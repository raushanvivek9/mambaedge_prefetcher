"""
MambaEdge Inference
====================
Loads mambaedge_best.pth and predicts the next memory access
(page index + offset index) for a given sequence of past accesses.

Usage examples
--------------
# 1. Predict from a live sequence of addresses (hex strings):
    from mambaedge_inference import MambaEdgeInference
    engine = MambaEdgeInference('/home/cs25m115/mamba/checkpoints/mambaedge_best.pth')
    pred_addr = engine.predict_address(['0x7f3a1000', '0x7f3a2000', ...])  # ≥32 addrs

# 2. Run on a full .xz trace file and get per-access predictions:
    results = engine.predict_trace('/path/to/trace.champsimtrace.xz')
    print(f"Address accuracy: {results['addr_accuracy']:.2f}%")

# 3. Single-step streaming (feed one address at a time):
    engine.reset()
    for addr in live_stream:
        pred = engine.step(addr)   # returns None until buffer has 32 entries
        if pred:
            print(f"Next prefetch → page={pred['page']} offset={pred['offset']} addr={pred['address']:#x}")
"""

import os
import math
import lzma
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants — must match CFG used during training ───────────────────────────
PAGE_BITS   = 12
OFFSET_BITS = 6
N_PAGES     = 512
N_OFFSETS   = 64
SEQ_LEN     = 32
D_MODEL     = 64
N_LAYERS    = 4
D_STATE     = 16
EXPAND      = 2

# ── GPU capability check (Triton needs sm_70+) ────────────────────────────────
def _triton_available() -> bool:
    if not torch.cuda.is_available():
        return False
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            print(f"[INFO] GPU sm_{major}{minor} < sm_70 — using pure-PyTorch SSM.")
            return False
    return True

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    MAMBA_TRITON = _triton_available()
except ImportError:
    MAMBA_TRITON = False

# ── Model (identical to training script) ─────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x, angles):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * torch.cos(angles) - x2 * torch.sin(angles),
                      x1 * torch.sin(angles) + x2 * torch.cos(angles)], dim=-1)


class Mamba3Block(nn.Module):
    def __init__(self, d_model=64, d_state=16, expand=2, chunk_size=128, norm_eps=1e-5):
        super().__init__()
        E = d_model * expand
        H, P = 1, E
        td = d_state // 2
        self.d_state = d_state
        self.E = E
        self.chunk_size = chunk_size
        self._td, self._H, self._P = td, H, P

        self.in_proj   = nn.Linear(d_model, E+E+d_state+d_state+H+H+td+H, bias=False)
        self.bc_norm_B = RMSNorm(d_state, norm_eps)
        self.bc_norm_C = RMSNorm(d_state, norm_eps)
        self.B_bias    = nn.Parameter(torch.ones(d_state))
        self.C_bias    = nn.Parameter(torch.ones(d_state))

        dt_i = torch.exp(
            torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.register_buffer('dt_bias', dt_i + torch.log(-torch.expm1(-dt_i)))
        self.A_log    = nn.Parameter(torch.log(0.5 * torch.ones(H)))
        self.out_proj = nn.Linear(E, d_model, bias=False)

    def _fallback_ssm(self, x_trap, dt, A, B_, C_):
        """Pure-PyTorch SSM scan for GPUs below sm_70."""
        B_sz, L, H, P = x_trap.shape
        N = B_.shape[-1]
        u   = x_trap.mean(dim=-1)                          # (B,L,H)
        dA  = torch.exp(dt * A.view(1, 1, H))              # (B,L,H)
        dB  = dt.unsqueeze(-1) * B_.unsqueeze(2)           # (B,L,H,N)
        dBu = dB * u.unsqueeze(-1)                         # (B,L,H,N)
        CHUNK = 256
        state = torch.zeros(B_sz, H, N, device=x_trap.device, dtype=x_trap.dtype)
        outs  = torch.empty(B_sz, L, H, device=x_trap.device, dtype=x_trap.dtype)
        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            for t in range(start, end):
                state = dA[:, t, :].unsqueeze(-1) * state + dBu[:, t]
                outs[:, t] = (state * C_[:, t].unsqueeze(1)).sum(-1)
        return outs  # (B,L,H)

    def forward(self, u):
        B, L, _ = u.shape
        N, H, P = self.d_state, self._H, self._P
        z = self.in_proj(u)
        ptr = 0

        def take(n):
            nonlocal ptr
            o = z[..., ptr:ptr+n]; ptr += n; return o

        x    = take(self.E);  gate = take(self.E)
        B_   = take(N);       C_   = take(N)
        dt_  = take(H);       A_   = take(H)
        th   = take(self._td); lam_ = take(H)

        x    = F.silu(x);    gate = F.silu(gate)
        dt   = F.softplus(dt_ + self.dt_bias)
        A    = -F.softplus(A_ + self.A_log)
        lam  = torch.sigmoid(lam_)
        alpha = torch.exp(dt * A)

        B_ = self.bc_norm_B(B_) + self.B_bias
        C_ = self.bc_norm_C(C_) + self.C_bias

        ca = torch.cumsum(dt.mean(-1, keepdim=True) * th, dim=1)
        B_ = apply_rope(B_, ca)
        C_ = apply_rope(C_, ca)

        xh = x.view(B, L, H, P)
        xp = torch.roll(xh, 1, dims=1); xp[:, 0] = 0.0
        lh = lam.unsqueeze(-1)
        xt = lh * xh + (1 - lh) * alpha.unsqueeze(-1) * xp

        if MAMBA_TRITON:
            y = mamba_chunk_scan_combined(
                xt, dt, A.mean(dim=(0, 1)),
                B_.unsqueeze(2), C_.unsqueeze(2),
                chunk_size=self.chunk_size,
                D=None, z=None, dt_bias=None, dt_softplus=False,
            )
            y = y.reshape(B, L, self.E)
        else:
            y_raw = self._fallback_ssm(xt, dt, A.mean(dim=(0, 1)), B_, C_)
            y = y_raw.reshape(B, L, H).expand(B, L, self.E).contiguous()

        return self.out_proj(y * gate)


class MambaEdgePrefetcher(nn.Module):
    def __init__(self, n_pages=N_PAGES, n_offsets=N_OFFSETS,
                 d_model=D_MODEL, n_layers=N_LAYERS,
                 d_state=D_STATE, expand=EXPAND, dropout=0.0):
        super().__init__()
        self.page_embed   = nn.Embedding(n_pages, d_model)
        self.offset_embed = nn.Embedding(n_offsets, d_model // 4)
        self.input_proj   = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.LayerNorm(d_model),
        )
        self.mamba_layers = nn.ModuleList([
            Mamba3Block(d_model=d_model, d_state=d_state, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.page_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_pages),
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, n_offsets),
        )

    def forward(self, pg, off):
        p = self.page_embed(pg)
        o = self.offset_embed(off)
        x = self.dropout(self.input_proj(torch.cat([p, o], dim=-1)))
        for m, n in zip(self.mamba_layers, self.norms):
            x = x + m(n(x))
        last = x[:, -1, :]
        return self.page_head(last), self.offset_head(last)


# ── Address encode/decode helpers ─────────────────────────────────────────────
def addr_to_page_offset(addr: int):
    """Convert raw memory address → (page_idx, offset_idx) matching training."""
    page   = (addr >> PAGE_BITS) % N_PAGES
    offset = ((addr & ((1 << PAGE_BITS) - 1)) >> (PAGE_BITS - OFFSET_BITS)) % N_OFFSETS
    return int(page), int(offset)

def page_offset_to_addr(page: int, offset: int) -> int:
    """Reconstruct address from page + offset indices (page<<12 | offset<<6)."""
    return (page << PAGE_BITS) | (offset << OFFSET_BITS)


# ── Main inference engine ─────────────────────────────────────────────────────
class MambaEdgeInference:
    """
    Drop-in inference engine for MambaEdge prefetcher.

    Parameters
    ----------
    checkpoint_path : str
        Path to mambaedge_best.pth  (or any .pth checkpoint)
    device : str or torch.device, optional
        'cuda', 'cpu', or 'auto' (default — picks GPU if available)
    batch_size : int
        Batch size for trace-level prediction (default 512)
    """

    def __init__(self, checkpoint_path: str, device='auto', batch_size=512):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size
        self.seq_len    = SEQ_LEN
        self._buffer    = collections.deque(maxlen=SEQ_LEN)  # for streaming

        self.model = self._load_model(checkpoint_path)
        print(f"[MambaEdge] Model ready on {self.device}  "
              f"| params: {sum(p.numel() for p in self.model.parameters()):,}  "
              f"| Triton SSM: {MAMBA_TRITON}")

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self, path: str) -> nn.Module:
        _raw = torch.load(path, map_location=self.device)
        # Training saves: {epoch, model_state, optimizer_state, cfg, ...}
        # Extract whichever key holds the actual weights
        if isinstance(_raw, dict):
            if 'model_state' in _raw:           # ← training script key
                state = _raw['model_state']
            elif 'model_state_dict' in _raw:
                state = _raw['model_state_dict']
            elif 'state_dict' in _raw:
                state = _raw['state_dict']
            else:
                state = _raw                    # raw state dict
        else:
            state = _raw
        # Strip DDP 'module.' prefix if saved from DDP wrapper
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model = MambaEdgePrefetcher().to(self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    # ── Core inference ────────────────────────────────────────────────────────
    @torch.no_grad()
    def _forward(self, pages_np: np.ndarray, offsets_np: np.ndarray):
        """
        Run model forward pass.

        Parameters
        ----------
        pages_np   : int64 ndarray of shape (B, SEQ_LEN)
        offsets_np : int64 ndarray of shape (B, SEQ_LEN)

        Returns
        -------
        pred_pages   : int ndarray (B,)
        pred_offsets : int ndarray (B,)
        page_logits  : float ndarray (B, N_PAGES)   — for top-k use
        """
        pg  = torch.from_numpy(pages_np.astype(np.int64)).to(self.device)
        off = torch.from_numpy(offsets_np.astype(np.int64)).to(self.device)
        pl, ol = self.model(pg, off)
        return (
            pl.argmax(-1).cpu().numpy().astype(int),
            ol.argmax(-1).cpu().numpy().astype(int),
            pl.cpu().numpy(),
        )

    # ── API 1: predict from a list of hex/int addresses ───────────────────────
    def predict_address(self, addresses, topk: int = 1) -> list:
        """
        Predict next prefetch address(es) from a list of past addresses.

        Parameters
        ----------
        addresses : list of int or hex-string, length >= SEQ_LEN (32)
            Recent memory access addresses in chronological order.
            If longer than SEQ_LEN, the last SEQ_LEN are used.
        topk : int
            Number of candidate prefetch addresses to return (default 1).

        Returns
        -------
        list of dicts, length = topk:
            [{'page': int, 'offset': int, 'address': int, 'prob': float}, ...]
        """
        if len(addresses) < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} addresses, got {len(addresses)}")

        # Parse hex strings if needed
        parsed = [int(a, 16) if isinstance(a, str) else int(a)
                  for a in addresses[-self.seq_len:]]

        pages   = np.array([addr_to_page_offset(a)[0] for a in parsed], dtype=np.int64)
        offsets = np.array([addr_to_page_offset(a)[1] for a in parsed], dtype=np.int64)

        pg  = torch.tensor(pages,   dtype=torch.long, device=self.device).unsqueeze(0)
        off = torch.tensor(offsets, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            pl, ol = self.model(pg, off)

        page_probs = torch.softmax(pl[0], dim=-1)
        off_probs  = torch.softmax(ol[0], dim=-1)

        top_pages = page_probs.topk(topk)
        results = []
        for i in range(topk):
            p_idx  = top_pages.indices[i].item()
            p_prob = top_pages.values[i].item()
            o_idx  = off_probs.argmax().item()
            results.append({
                'page':    p_idx,
                'offset':  o_idx,
                'address': page_offset_to_addr(p_idx, o_idx),
                'prob':    round(p_prob, 4),
            })
        return results

    # ── API 2: streaming (one address at a time) ──────────────────────────────
    def reset(self):
        """Clear the internal address buffer (call before a new trace stream)."""
        self._buffer.clear()

    def step(self, address) -> dict | None:
        """
        Feed one address and get a prefetch prediction.

        Returns None until the buffer has SEQ_LEN (32) entries.
        After that returns a dict on every call.

        Parameters
        ----------
        address : int or hex-string

        Returns
        -------
        dict with keys: page, offset, address, or None
        """
        addr = int(address, 16) if isinstance(address, str) else int(address)
        pg, off = addr_to_page_offset(addr)
        self._buffer.append((pg, off))

        if len(self._buffer) < self.seq_len:
            return None

        pages   = np.array([x[0] for x in self._buffer], dtype=np.int64)
        offsets = np.array([x[1] for x in self._buffer], dtype=np.int64)
        pred_pages, pred_offs, _ = self._forward(pages[None], offsets[None])

        p, o = int(pred_pages[0]), int(pred_offs[0])
        return {
            'page':    p,
            'offset':  o,
            'address': page_offset_to_addr(p, o),
        }

    # ── API 3: full trace file ─────────────────────────────────────────────────
    def predict_trace(self, trace_path: str, max_accesses: int = 700_000) -> dict:
        """
        Run inference over an entire .xz ChampSim trace file.

        Returns a results dict with accuracy metrics.
        """
        name = os.path.basename(trace_path)
        print(f"\n[MambaEdge] Predicting trace: {name}")
        t0 = time.time()

        # ── Load trace ────────────────────────────────────────────────────────
        print("  Loading trace ...", end=" ", flush=True)
        addrs = []
        with lzma.open(trace_path, 'rt', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    addr = int(parts[2], 16) if len(parts) >= 3 else int(parts[0], 16)
                    addrs.append(addr)
                except Exception:
                    continue
                if len(addrs) >= max_accesses:
                    break

        addrs   = np.array(addrs, dtype=np.int64)
        pages   = ((addrs >> PAGE_BITS) % N_PAGES).astype(np.int16)
        offsets = (((addrs & ((1 << PAGE_BITS) - 1)) >> (PAGE_BITS - OFFSET_BITS))
                   % N_OFFSETS).astype(np.int8)
        N = len(pages)
        print(f"{N:,} accesses loaded in {time.time()-t0:.1f}s")

        # ── Batched inference ─────────────────────────────────────────────────
        print("  Running inference ...", end=" ", flush=True)
        t1 = time.time()
        pred_pg  = np.zeros(N, dtype=np.int16)
        pred_off = np.zeros(N, dtype=np.int8)

        for start in range(self.seq_len, N, self.batch_size):
            end = min(start + self.batch_size, N)
            B   = end - start
            pg_win  = np.zeros((B, self.seq_len), dtype=np.int64)
            off_win = np.zeros((B, self.seq_len), dtype=np.int64)
            for j in range(B):
                i = start + j
                pg_win[j]  = pages  [i - self.seq_len:i]
                off_win[j] = offsets[i - self.seq_len:i]
            pp, po, _ = self._forward(pg_win, off_win)
            pred_pg [start:end] = pp
            pred_off[start:end] = po

        infer_time = time.time() - t1
        print(f"done in {infer_time:.1f}s")

        # ── Metrics ───────────────────────────────────────────────────────────
        valid = slice(self.seq_len, N)
        true_pg  = pages  [valid].astype(int)
        true_off = offsets[valid].astype(int)
        p_pred   = pred_pg [valid].astype(int)
        o_pred   = pred_off[valid].astype(int)

        page_correct = (p_pred == true_pg)
        off_correct  = (o_pred == true_off)
        addr_correct = page_correct & off_correct

        page_acc = page_correct.mean() * 100
        off_acc  = off_correct.mean()  * 100
        addr_acc = addr_correct.mean() * 100
        total    = time.time() - t0

        print(f"\n  ── Results: {name} ──")
        print(f"  Page  accuracy (exact) : {page_acc:>7.2f}%")
        print(f"  Offset accuracy (exact): {off_acc:>7.2f}%")
        print(f"  Address accuracy       : {addr_acc:>7.2f}%  ← page+offset both correct")
        print(f"  Total time             : {total:.1f}s  "
              f"({(N - self.seq_len) / infer_time:,.0f} predictions/sec)")

        return {
            'trace':      name,
            'n_accesses': N,
            'page_acc':   round(page_acc, 4),
            'offset_acc': round(off_acc,  4),
            'addr_acc':   round(addr_acc, 4),
            'infer_time': round(infer_time, 2),
            'pred_pages':   pred_pg,
            'pred_offsets': pred_off,
            'true_pages':   pages,
            'true_offsets': offsets,
        }


# ── Quick demo when run directly ──────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    CKPT = '/home/cs25m115/mamba/checkpoints/mambaedge_best.pth'
    if not os.path.exists(CKPT):
        print(f"[ERROR] Checkpoint not found: {CKPT}")
        print("  Edit CKPT path at the bottom of this file.")
        sys.exit(1)

    engine = MambaEdgeInference(CKPT)

    # ── Demo 1: predict from synthetic address sequence ───────────────────────
    print("\n── Demo 1: predict from address list ──")
    fake_addrs = [0x7f000000 + i * 0x1000 for i in range(40)]
    preds = engine.predict_address(fake_addrs, topk=3)
    print(f"  Input : last 32 of {len(fake_addrs)} addresses")
    for rank, p in enumerate(preds, 1):
        print(f"  Top-{rank}: page={p['page']:>3}  offset={p['offset']:>2}  "
              f"addr={p['address']:#010x}  prob={p['prob']:.4f}")

    # ── Demo 2: streaming step-by-step ───────────────────────────────────────
    print("\n── Demo 2: streaming (step by step) ──")
    engine.reset()
    for i, addr in enumerate(fake_addrs):
        result = engine.step(addr)
        if result:
            print(f"  step {i:>3}: → prefetch {result['address']:#010x}  "
                  f"(page={result['page']}, offset={result['offset']})")
            if i >= 35:
                break  # just show a few

    # ── Demo 3: full trace file (pass path as argument) ───────────────────────
    if len(sys.argv) > 1:
        trace_path = sys.argv[1]
        print(f"\n── Demo 3: full trace {os.path.basename(trace_path)} ──")
        results = engine.predict_trace(trace_path)