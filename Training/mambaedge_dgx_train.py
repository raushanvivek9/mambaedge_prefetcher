"""
MambaEdge DGX Multi-GPU Training Script
========================================
Recommended hyperparameters (from summary):
  Epochs         : 150
  Batch size     : 512  (per-GPU; effective = 512 × N_GPUS)
  Stride         : 2
  LR             : 1e-3
  LR schedule    : 10-epoch warmup + CosineAnnealing
  Label smooth   : 0.1 (page),  0.05 (offset)
  Early stopping : patience=20

Usage (single node, all GPUs):
  torchrun --nproc_per_node=NUM_GPUS mambaedge_dgx_train.py

Usage (SLURM / DGX SuperPOD):
  srun --ntasks-per-node=NUM_GPUS python mambaedge_dgx_train.py

Checkpoints saved every epoch as:
  checkpoints/mambaedge_epoch{N}.pth   ← resume-safe checkpoint (all state)
  checkpoints/mambaedge_best.pth       ← best val-loss model weights only
  checkpoints/mambaedge_final.pt       ← final exported weights (inference)
"""

import os
import lzma
import time
import math
import json
import hashlib
import collections
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

def _gpu_supports_triton_mamba() -> bool:
    """
    mamba_ssm Triton kernels use PTX '.acq_rel' which requires sm_70+.
    Pascal GPUs (sm_61, e.g. GTX 1080 / P100) crash at first backward pass:
        ptxas error: Feature '.acq_rel' requires .target sm_70 or higher
    Check every visible GPU and return False if any is below sm_70.
    """
    if not torch.cuda.is_available():
        return False
    for i in range(torch.cuda.device_count()):
        major, _minor = torch.cuda.get_device_capability(i)
        if major < 7:
            print(f"[WARN] GPU {i} ({torch.cuda.get_device_name(i)}) "
                  f"is sm_{major}{_minor} — below sm_70 minimum for "
                  f"mamba_ssm Triton kernels. Forcing pure-PyTorch fallback.")
            return False
    return True

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    MAMBA_AVAILABLE = _gpu_supports_triton_mamba()
    if not MAMBA_AVAILABLE:
        print("[WARN] mamba_ssm disabled — GPU is sm_61 (need sm_70+). "
              "Using pure-PyTorch SSM fallback (same accuracy, slightly slower).")
except ImportError:
    MAMBA_AVAILABLE = False
    print("[WARN] mamba_ssm not found — using pure-PyTorch SSM fallback.")

# ════════════════════════════════════════════════════════════════════════════════
# ── 0. Distributed setup helpers ────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def setup_ddp():
    """
    Initialise DDP. Works with:
      - torchrun --standalone --nproc_per_node=N  (recommended)
      - SLURM srun (sets RANK/LOCAL_RANK/WORLD_SIZE automatically)
      - Plain python (falls back to single-GPU)
    """
    # torchrun sets RANK; srun may leave some vars empty — fill all defaults
    rank_str       = os.environ.get("RANK",       "").strip()
    local_rank_str = os.environ.get("LOCAL_RANK", "").strip()
    world_size_str = os.environ.get("WORLD_SIZE", "").strip()

    # If any key var is missing or empty, default to single-process
    if not rank_str or not local_rank_str or not world_size_str:
        os.environ["RANK"]        = "0"
        os.environ["LOCAL_RANK"]  = "0"
        os.environ["WORLD_SIZE"]  = "1"

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),   # fixes "unknown GPU" warning / hang
        timeout=datetime.timedelta(seconds=3600),        # 1-hour timeout (trace loading can take >10 min)
    )
    return local_rank, dist.get_world_size(), dist.get_rank()


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def log(rank, *args, **kwargs):
    """Print only from rank-0."""
    if rank == 0:
        print(*args, **kwargs)


# ════════════════════════════════════════════════════════════════════════════════
# ── 1. Hyperparameters ──────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

CFG = dict(
    # ── Data
    trace_dir        = "/home/cs25m115/spec2017",
    max_per_trace    = 2000000,
    seq_len          = 32,
    stride           = 2,          # ← recommended: 2 (was 4)
    train_frac       = 0.8,
    val_frac         = 0.1,

    # ── Training
    epochs           = 150,        # ← recommended: 150 (was 50)
    batch_size       = 512,        # ← per-GPU; effective = 512 × N_GPUS (was 256)
    lr               = 1e-3,       # ← recommended: 1e-3 (was 3e-3)
    weight_decay     = 1e-4,
    grad_clip        = 1.0,
    warmup_epochs    = 10,         # ← NEW: linear warmup then cosine
    eta_min          = 1e-6,
    label_smooth_pg  = 0.1,        # ← NEW: label smoothing page head
    label_smooth_off = 0.05,       # ← NEW: label smoothing offset head
    early_stop_pat   = 20,         # ← NEW: patience=20

    # ── Model
    n_pages          = 512,
    n_offsets        = 64,
    d_model          = 64,
    n_layers         = 4,
    d_state          = 16,
    expand           = 2,
    dropout          = 0.1,

    # ── IPEX throttle
    throttle_thresh  = 0.50,
    recover_thresh   = 0.70,
    window_size      = 10,
    throttle_pat     = 3,
    recover_pat      = 5,

    # ── I/O
    ckpt_dir         = "./checkpoints",
    num_workers      = 4,
)

# Trace files (edit paths as needed on your DGX)
# Auto-discover all .xz trace files in trace_dir (sorted for reproducibility)
TRACE_PATHS = sorted([
    os.path.join(CFG["trace_dir"], f)
    for f in os.listdir(CFG["trace_dir"])
    if f.endswith(".xz")
])
print(f"[INFO] Found {len(TRACE_PATHS)} trace file(s) in {CFG['trace_dir']}")

# ════════════════════════════════════════════════════════════════════════════════
# ── 2. Data loading ─────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

PAGE_BITS   = 12
OFFSET_BITS = 6


def parse_line_fast(line):
    parts = line.split()
    if not parts:
        return None
    try:
        token = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else parts[0])
        return int(token, 16)
    except Exception:
        return None


def process_trace_to_cache(path, max_accesses=2000000):
    fname       = os.path.basename(path)
    cache_pages = path + ".pages.npy"
    cache_offs  = path + ".offs.npy"

    if os.path.exists(cache_pages) and os.path.exists(cache_offs):
        pages = np.load(cache_pages)
        offs  = np.load(cache_offs)
        print(f"[CACHE] {fname}: {len(pages):,} accesses | {len(np.unique(pages))} unique pages")
        return pages, offs

    addrs = np.empty(max_accesses, dtype=np.int64)
    count = 0
    with lzma.open(path, "rt", errors="ignore") as f:
        for line in tqdm(f, desc=fname, leave=False):
            addr = parse_line_fast(line)
            if addr is None:
                continue
            addrs[count] = addr
            count += 1
            if count >= max_accesses:
                break

    addrs = addrs[:count]
    if count == 0:
        print(f"[NEW] {fname}: 0 accesses — skipped")
        return None, None

    N   = CFG["n_pages"]
    OFF = CFG["n_offsets"]
    pages = ((addrs >> PAGE_BITS) % N).astype(np.int16)
    offs  = (((addrs & ((1 << PAGE_BITS)-1)) >> (PAGE_BITS - OFFSET_BITS)) % OFF).astype(np.int8)

    # Write atomically: np.save() always appends ".npy" to the filename,
    # so the temp files land as  <cache>.tmp.npy  — rename those to <cache>.
    np.save(cache_pages + ".tmp", pages)   # → cache_pages + ".tmp.npy"
    np.save(cache_offs  + ".tmp", offs)    # → cache_offs  + ".tmp.npy"
    os.replace(cache_pages + ".tmp.npy", cache_pages)
    os.replace(cache_offs  + ".tmp.npy", cache_offs)
    print(f"[NEW] {fname}: {len(pages):,} accesses | {len(np.unique(pages))} unique pages")
    return pages, offs


def load_all_traces(trace_paths, max_per_trace=2000000):
    all_pages, all_offs = [], []
    for p in trace_paths:
        if not os.path.exists(p):
            print(f"[SKIP] {p} not found")
            continue
        pg, off = process_trace_to_cache(p, max_per_trace)
        if pg is not None:
            all_pages.append(pg)
            all_offs.append(off)
    if not all_pages:
        raise RuntimeError("No trace files loaded — check TRACE_PATHS!")
    return np.concatenate(all_pages), np.concatenate(all_offs)


class TraceDataset(Dataset):
    def __init__(self, pages, offsets, seq_len=32, stride=1):
        self.pages   = torch.from_numpy(pages.astype(np.int64))
        self.offsets = torch.from_numpy(offsets.astype(np.int64))
        self.seq_len = seq_len
        self.indices = list(range(0, len(pages) - seq_len, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.seq_len
        return (self.pages[s:e], self.offsets[s:e],
                self.pages[e],   self.offsets[e])


def make_dataloaders(pages, offsets, rank, world_size):
    N  = len(pages)
    t1 = int(N * CFG["train_frac"])
    t2 = int(N * (CFG["train_frac"] + CFG["val_frac"]))

    train_ds = TraceDataset(pages[:t1],   offsets[:t1],   CFG["seq_len"], CFG["stride"])
    val_ds   = TraceDataset(pages[t1:t2], offsets[t1:t2], CFG["seq_len"], stride=1)
    test_ds  = TraceDataset(pages[t2:],   offsets[t2:],   CFG["seq_len"], stride=1)

    if rank == 0:
        print(f"Dataset sizes — Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    # DistributedSampler splits data evenly across all GPUs
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                       rank=rank, shuffle=False, drop_last=False)

    kw = dict(
        batch_size  = CFG["batch_size"],
        num_workers = CFG["num_workers"],
        pin_memory  = True,
        persistent_workers = (CFG["num_workers"] > 0),
    )
    train_dl = DataLoader(train_ds, sampler=train_sampler, **kw)
    val_dl   = DataLoader(val_ds,   sampler=val_sampler,   **kw)
    # Test runs on rank-0 only — no sampler needed
    test_dl  = DataLoader(test_ds, batch_size=CFG["batch_size"],
                          shuffle=False, num_workers=CFG["num_workers"],
                          pin_memory=True)
    return train_dl, val_dl, test_dl, train_sampler


# ════════════════════════════════════════════════════════════════════════════════
# ── 3. Model ────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x, angles):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat([x1*torch.cos(angles) - x2*torch.sin(angles),
                      x1*torch.sin(angles) + x2*torch.cos(angles)], dim=-1)


class Mamba3Block(nn.Module):
    """
    Mamba-3 SSM block (arXiv:2603.15569).
    Features: Exponential-Trapezoidal discretization, Data-dependent RoPE,
    BCNorm + learnable B/C biases, no short causal conv.
    """
    def __init__(self, d_model=128, d_state=64, expand=2,
                 dt_min=0.001, dt_max=0.1, chunk_size=128, norm_eps=1e-5):
        super().__init__()
        E          = d_model * expand
        H, P       = 1, E
        theta_dim  = d_state // 2
        self.d_state, self.E, self.chunk_size = d_state, E, chunk_size
        self._theta_dim, self._H, self._P     = theta_dim, H, P

        self.in_proj   = nn.Linear(d_model, E+E+d_state+d_state+H+H+theta_dim+H, bias=False)
        self.bc_norm_B = RMSNorm(d_state, norm_eps)
        self.bc_norm_C = RMSNorm(d_state, norm_eps)
        self.B_bias    = nn.Parameter(torch.ones(d_state))
        self.C_bias    = nn.Parameter(torch.ones(d_state))

        dt_init = torch.exp(
            torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        self.register_buffer("dt_bias", dt_init + torch.log(-torch.expm1(-dt_init)))
        self.A_log    = nn.Parameter(torch.log(0.5 * torch.ones(H)))
        self.out_proj = nn.Linear(E, d_model, bias=False)

    def _fallback_ssm(self, x_trap, dt, A, B_, C_):
        """
        Pure-PyTorch SSM scan — used when GPU compute capability < sm_70.

        Implements the discrete SSM recurrence in chunked form for memory
        efficiency.  Matches the mamba_chunk_scan_combined interface:

          h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * (x_trap_t averaged over P)
          y_t = C_t · h_t                       shape: (B, L, H)

        Args:
          x_trap : (B, L, H, P)
          dt     : (B, L, H)
          A      : (H,)          — scalar per head (already .mean'd by caller)
          B_     : (B, L, N)
          C_     : (B, L, N)
        Returns:
          y      : (B, L, H)
        """
        B_sz, L, H, P = x_trap.shape
        N = B_.shape[-1]
        dtype = x_trap.dtype

        # Reduce x over the P (expand) dimension → (B, L, H)
        u = x_trap.mean(dim=-1)                           # (B, L, H)

        # dA: (B, L, H),  dB: (B, L, H, N)
        dA = torch.exp(dt * A.view(1, 1, H))              # (B, L, H)
        # Outer product of dt and B_ for each head:
        # dt: (B,L,H) → (B,L,H,1),  B_: (B,L,N) → (B,L,1,N)
        dB = dt.unsqueeze(-1) * B_.unsqueeze(2)           # (B, L, H, N)
        # Scale by input u: (B,L,H) → (B,L,H,1)
        dBu = dB * u.unsqueeze(-1)                        # (B, L, H, N)

        # Sequential scan — chunked in time to keep memory bounded
        CHUNK = 256
        state = torch.zeros(B_sz, H, N, device=x_trap.device, dtype=dtype)
        outs  = torch.empty(B_sz, L, H, device=x_trap.device, dtype=dtype)

        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            for t in range(start, end):
                state = dA[:, t, :].unsqueeze(-1) * state + dBu[:, t]  # (B,H,N)
                # y_t = sum_n  C_t[n] * state[h,n]
                # C_: (B,L,N) → (B,1,N) for this t
                outs[:, t] = (state * C_[:, t].unsqueeze(1)).sum(-1)   # (B,H)

        return outs  # (B, L, H)

    def forward(self, u):
        B, L, _ = u.shape
        N, H, P = self.d_state, self._H, self._P
        z   = self.in_proj(u)
        ptr = 0

        def take(n):
            nonlocal ptr
            out  = z[..., ptr:ptr+n]
            ptr += n
            return out

        x       = take(self.E)
        gate    = take(self.E)
        B_      = take(N)
        C_      = take(N)
        dt_     = take(H)
        A_log_  = take(H)
        theta_  = take(self._theta_dim)
        lam_    = take(H)

        x    = F.silu(x)
        gate = F.silu(gate)
        dt   = F.softplus(dt_ + self.dt_bias)
        A    = -F.softplus(A_log_ + self.A_log)
        lam  = torch.sigmoid(lam_)
        alpha = torch.exp(dt * A)

        B_ = self.bc_norm_B(B_) + self.B_bias
        C_ = self.bc_norm_C(C_) + self.C_bias

        dangle   = dt.mean(-1, keepdim=True) * theta_
        cumangle = torch.cumsum(dangle, dim=1)
        B_ = apply_rope(B_, cumangle)
        C_ = apply_rope(C_, cumangle)

        x_h    = x.view(B, L, H, P)
        x_prev = torch.roll(x_h, 1, dims=1)
        x_prev[:, 0] = 0.0
        lam_hp = lam.unsqueeze(-1)
        x_trap = lam_hp * x_h + (1 - lam_hp) * alpha.unsqueeze(-1) * x_prev

        if MAMBA_AVAILABLE:
            y = mamba_chunk_scan_combined(
                x_trap, dt, A.mean(dim=(0, 1)),
                B_.unsqueeze(2), C_.unsqueeze(2),
                chunk_size=self.chunk_size,
                D=None, z=None, dt_bias=None, dt_softplus=False,
            )
            y = y.reshape(B, L, self.E)
        else:
            y_raw = self._fallback_ssm(x_trap, dt, A.mean(dim=(0, 1)), B_, C_)
            y = y_raw.reshape(B, L, H).expand(B, L, self.E).contiguous()

        return self.out_proj(y * gate)


class MambaEdgePrefetcher(nn.Module):
    def __init__(self, n_pages=512, n_offsets=64,
                 d_model=128, n_layers=4, d_state=64, expand=2, dropout=0.1):
        super().__init__()
        self.page_embed   = nn.Embedding(n_pages,   d_model)
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
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_pages),
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, n_offsets),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, page_seq, offset_seq):
        p = self.page_embed(page_seq)
        o = self.offset_embed(offset_seq)
        x = self.input_proj(torch.cat([p, o], dim=-1))
        x = self.dropout(x)
        for mamba, norm in zip(self.mamba_layers, self.norms):
            x = x + mamba(norm(x))
        last = x[:, -1, :]
        return self.page_head(last), self.offset_head(last)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════════════════
# ── 4. IPEX Throttle ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

POWER_HIGH = "HIGH"
POWER_MED  = "MED"
POWER_LOW  = "LOW"
DEGREE_MAP = {POWER_HIGH: 8, POWER_MED: 4, POWER_LOW: 2}


class IPEXThrottle:
    def __init__(self, throttle_threshold=0.50, recover_threshold=0.70,
                 window_size=10, throttle_patience=3, recover_patience=5, verbose=True):
        self.throttle_threshold = throttle_threshold
        self.recover_threshold  = recover_threshold
        self.window_size        = window_size
        self.throttle_patience  = throttle_patience
        self.recover_patience   = recover_patience
        self.verbose            = verbose
        self._state             = POWER_HIGH
        self._acc_window        = collections.deque(maxlen=window_size)
        self._below_count = self._above_count = self._batch_idx = 0
        self.history = {"accuracy": [], "prefetch_degree": [], "state": [], "events": []}

    @property
    def state(self):          return self._state
    @property
    def prefetch_degree(self): return DEGREE_MAP[self._state]
    @property
    def rolling_accuracy(self):
        return sum(self._acc_window) / len(self._acc_window) if self._acc_window else 0.0

    def update(self, accuracy: float) -> int:
        self._acc_window.append(accuracy)
        self._batch_idx += 1
        avg = self.rolling_accuracy
        self.history["accuracy"].append(round(avg, 4))
        self.history["prefetch_degree"].append(self.prefetch_degree)
        self.history["state"].append(self._state)
        if avg < self.throttle_threshold:
            self._below_count += 1; self._above_count = 0
            if self._below_count >= self.throttle_patience:
                self._step_down(); self._below_count = 0
        elif avg > self.recover_threshold:
            self._above_count += 1; self._below_count = 0
            if self._above_count >= self.recover_patience:
                self._step_up(); self._above_count = 0
        else:
            self._below_count = self._above_count = 0
        return self.prefetch_degree

    def _step_down(self):
        old = self._state
        if self._state == POWER_HIGH: self._state = POWER_MED
        elif self._state == POWER_MED: self._state = POWER_LOW
        if old != self._state:
            ev = f"THROTTLE {old}→{self._state} degree {DEGREE_MAP[old]}→{self.prefetch_degree}x acc={self.rolling_accuracy*100:.1f}%"
            self.history["events"].append((self._batch_idx, ev))
            if self.verbose: print(f"\n  [IPEX] ⬇  {ev}")

    def _step_up(self):
        old = self._state
        if self._state == POWER_LOW:  self._state = POWER_MED
        elif self._state == POWER_MED: self._state = POWER_HIGH
        if old != self._state:
            ev = f"RECOVER  {old}→{self._state} degree {DEGREE_MAP[old]}→{self.prefetch_degree}x acc={self.rolling_accuracy*100:.1f}%"
            self.history["events"].append((self._batch_idx, ev))
            if self.verbose: print(f"\n  [IPEX] ⬆  {ev}")

    def summary(self):
        print(f"\n{'='*52}\n  IPEX Throttle Summary\n{'='*52}")
        print(f"  Current state    : {self._state}")
        print(f"  Prefetch degree  : {self.prefetch_degree}x")
        print(f"  Rolling accuracy : {self.rolling_accuracy*100:.1f}%")
        print(f"  Total batches    : {self._batch_idx}")
        print(f"  Throttle events  :")
        for b, ev in (self.history["events"] or [(None, "(none)")]):
            print(f"    {('batch '+str(b)) if b else ''} : {ev}")
        print("="*52)


# ════════════════════════════════════════════════════════════════════════════════
# ── 5. Metrics ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def accuracy(logits, targets):
    return (logits.argmax(-1) == targets).float().mean().item()

def topk_accuracy(logits, targets, k=3):
    topk = logits.topk(k, dim=-1).indices
    return (topk == targets.unsqueeze(-1)).any(-1).float().mean().item()

def combined_addr_acc(pred_page, pred_offset, true_page, true_offset):
    pred_addr = (pred_page.long() << 12) | (pred_offset.long() << 6)
    true_addr = (true_page.long() << 12) | (true_offset.long() << 6)
    return (pred_addr == true_addr).float().mean().item()


# ════════════════════════════════════════════════════════════════════════════════
# ── 6. Checkpoint helpers ───────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, history,
                    ckpt_dir, is_best=False, rank=0):
    """Save a full resume-safe checkpoint (.pth) and optionally a best-model copy."""
    if rank != 0:
        return  # Only rank-0 writes checkpoints

    os.makedirs(ckpt_dir, exist_ok=True)

    # Unwrap DDP to get raw model state_dict
    raw_model = model.module if hasattr(model, "module") else model

    state = {
        "epoch":          epoch,
        "model_state":    raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss":  best_val_loss,
        "history":        history,
        "cfg":            CFG,
    }

    # Per-epoch checkpoint (keep last 3 to save disk)
    pth_path = os.path.join(ckpt_dir, f"mambaedge_epoch{epoch:04d}.pth")
    torch.save(state, pth_path)

    # Always keep a "latest" symlink for easy resuming
    latest_path = os.path.join(ckpt_dir, "mambaedge_latest.pth")
    if os.path.islink(latest_path) or os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.abspath(pth_path), latest_path)

    # Delete older checkpoints (keep last 3)
    all_ckpts = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.startswith("mambaedge_epoch") and f.endswith(".pth")
    ])
    for old in all_ckpts[:-3]:
        try:
            os.remove(os.path.join(ckpt_dir, old))
        except Exception:
            pass

    # Best model weights only
    if is_best:
        best_path = os.path.join(ckpt_dir, "mambaedge_best.pth")
        torch.save(state, best_path)
        print(f"  [CKPT] ★ New best saved → {best_path}")

    print(f"  [CKPT] Epoch {epoch} checkpoint → {pth_path}")


import copy
import os
import torch

def save_final_pt(model, ckpt_dir, rank=0):
    """Export final model as CPU TorchScript .pt file for ChampSim."""
    if rank != 0:
        return

    raw_model = model.module if hasattr(model, "module") else model

    # If your model has a Triton/Mamba GPU path, disable it for CPU export.
    if "MAMBA_TRITON" in globals():
        globals()["MAMBA_TRITON"] = False

    cpu_model = copy.deepcopy(raw_model).to("cpu")
    cpu_model.eval()

    example_pg = torch.zeros(1, 32, dtype=torch.long, device="cpu")
    example_off = torch.zeros(1, 32, dtype=torch.long, device="cpu")

    pt_path = os.path.join(ckpt_dir, "mambaedge_final_cpu_jit.pt")

    with torch.no_grad():
        traced = torch.jit.trace(cpu_model, (example_pg, example_off), strict=False)
        traced = torch.jit.freeze(traced)
        torch.jit.save(traced, pt_path)

    print(f"  [EXPORT] CPU TorchScript model -> {pt_path}")



def load_checkpoint(ckpt_path, model, optimizer, scheduler, device):
    """Load a checkpoint for resuming. Returns the start epoch."""
    print(f"  [RESUME] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])

    start_epoch  = state["epoch"] + 1
    best_val_loss = state.get("best_val_loss", float("inf"))
    history       = state.get("history", {})
    print(f"  [RESUME] Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss, history


# ════════════════════════════════════════════════════════════════════════════════
# ── 7. Main training loop ───────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def main():
    # ── Determine rank identity BEFORE init_process_group ────────────────────
    # torchrun exports LOCAL_RANK/RANK before launching, so we can read them.
    _rank_pre = int(os.environ.get("RANK", "0").strip() or "0")

    # ── Load / cache traces BEFORE init_process_group ────────────────────────
    # KEY FIX: .xz decompression can take 20+ minutes per file.
    # Any dist.barrier() during that time causes the NCCL watchdog on the
    # waiting ranks to fire its timeout and SIGABRT the job.
    #
    # Solution: every rank calls load_all_traces() independently, BEFORE
    # any distributed call is made, so no barrier is ever needed.
    #
    # First-run concurrency: process_trace_to_cache() checks for existing
    # .npy files and returns immediately if found.  All ranks race to write
    # the same files — last writer wins, all produce identical arrays.
    # Subsequent runs: .npy caches exist, so loading takes milliseconds.
    if _rank_pre == 0:
        print("[rank0] Pre-loading / caching traces before DDP init ...", flush=True)
    pages, offsets = load_all_traces(TRACE_PATHS, CFG["max_per_trace"])
    if _rank_pre == 0:
        print(f"[rank0] Traces ready ({len(pages):,} accesses). Initialising DDP ...", flush=True)

    # ── Now safe to initialise the process group (no I/O waits after this) ──
    local_rank, world_size, rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    log(rank, "="*68)
    log(rank, f"   MambaEdge — DGX Multi-GPU Training  ({world_size} GPU(s))")
    log(rank, "="*68)
    log(rank, f"   Epochs        : {CFG['epochs']}")
    log(rank, f"   Batch/GPU     : {CFG['batch_size']}  →  Effective: {CFG['batch_size']*world_size}")
    log(rank, f"   LR            : {CFG['lr']}  (warmup {CFG['warmup_epochs']} ep + cosine)")
    log(rank, f"   Stride        : {CFG['stride']}")
    log(rank, f"   Label smooth  : page={CFG['label_smooth_pg']}, offset={CFG['label_smooth_off']}")
    log(rank, f"   Early stop    : patience={CFG['early_stop_pat']}")
    log(rank, f"   Accesses      : {len(pages):,}  |  Unique pages: {len(np.unique(pages))}/{CFG['n_pages']}")
    log(rank, "="*68)

    train_dl, val_dl, test_dl, train_sampler = make_dataloaders(pages, offsets, rank, world_size)

    # ── Model ────────────────────────────────────────────────────────────────
    arch_args = dict(
        n_pages  = CFG["n_pages"],
        n_offsets= CFG["n_offsets"],
        d_model  = CFG["d_model"],
        n_layers = CFG["n_layers"],
        d_state  = CFG["d_state"],
        expand   = CFG["expand"],
        dropout  = CFG["dropout"],
    )
    model = MambaEdgePrefetcher(**arch_args).to(device)
    log(rank, f"   Model params  : {model.count_params():,}  ({model.count_params()*4/1e6:.1f} MB)")

    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    # ── Loss (with label smoothing) ──────────────────────────────────────────
    page_crit   = nn.CrossEntropyLoss(label_smoothing=CFG["label_smooth_pg"])
    offset_crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smooth_off"])

    # ── Optimiser ───────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    # ── Scheduler: linear warmup → CosineAnnealing ──────────────────────────
    warmup_sched = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0,
                            total_iters=CFG["warmup_epochs"])
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=CFG["epochs"] - CFG["warmup_epochs"],
                                     eta_min=CFG["eta_min"])
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_sched, cosine_sched],
                             milestones=[CFG["warmup_epochs"]])

    # ── IPEX Throttle ────────────────────────────────────────────────────────
    throttle = IPEXThrottle(
        throttle_threshold=CFG["throttle_thresh"],
        recover_threshold =CFG["recover_thresh"],
        window_size       =CFG["window_size"],
        throttle_patience =CFG["throttle_pat"],
        recover_patience  =CFG["recover_pat"],
        verbose           =(rank == 0),
    )

    # ── Resume from checkpoint if available ──────────────────────────────────
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)
    latest_ckpt = os.path.join(CFG["ckpt_dir"], "mambaedge_latest.pth")
    start_epoch   = 1
    best_val_loss = float("inf")
    history = {k: [] for k in ["tl","vl","tpa","vpa","toa","voa","vaa","degree","power_state","lr"]}

    if os.path.exists(latest_ckpt):
        start_epoch, best_val_loss, history = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, device)
    else:
        log(rank, "   No checkpoint found — training from scratch.")

    no_improve = 0  # for early stopping

    # ── Training header ──────────────────────────────────────────────────────
    log(rank, f"\n{'Ep':>4} {'TrainLoss':>10} {'ValLoss':>9} {'ValPageAcc':>11} "
              f"{'ValAddrAcc':>11} {'Degree':>7} {'State':>6} {'LR':>9} {'Time':>6}")
    log(rank, "─"*80)

    for epoch in range(start_epoch, CFG["epochs"] + 1):
        t0 = time.time()

        # ── Set epoch for DistributedSampler (reshuffles each epoch) ─────────
        train_sampler.set_epoch(epoch)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tl = tpa = toa = taa = 0.0

        for Xpb, Xob, ypb, yob in train_dl:
            Xpb = Xpb.to(device, non_blocking=True)
            Xob = Xob.to(device, non_blocking=True)
            ypb = ypb.to(device, non_blocking=True)
            yob = yob.to(device, non_blocking=True)

            pl, ol = model(Xpb, Xob)
            loss   = 0.7 * page_crit(pl, ypb) + 0.3 * offset_crit(ol, yob)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()

            tl  += loss.item()
            tpa += accuracy(pl, ypb)
            toa += accuracy(ol, yob)
            taa += combined_addr_acc(pl.argmax(-1), ol.argmax(-1), ypb, yob)

        nb  = len(train_dl)
        tl /= nb; tpa /= nb; toa /= nb; taa /= nb
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        vl = vpa = voa = vaa = 0.0

        with torch.no_grad():
            for Xpb, Xob, ypb, yob in val_dl:
                Xpb = Xpb.to(device, non_blocking=True)
                Xob = Xob.to(device, non_blocking=True)
                ypb = ypb.to(device, non_blocking=True)
                yob = yob.to(device, non_blocking=True)

                pl, ol = model(Xpb, Xob)
                vl  += (0.7 * page_crit(pl, ypb) + 0.3 * offset_crit(ol, yob)).item()
                vpa += accuracy(pl, ypb)
                voa += accuracy(ol, yob)
                batch_addr_acc = combined_addr_acc(pl.argmax(-1), ol.argmax(-1), ypb, yob)
                vaa += batch_addr_acc
                if rank == 0:
                    throttle.update(batch_addr_acc)

        nb  = len(val_dl)
        vl /= nb; vpa /= nb; voa /= nb; vaa /= nb

        # ── Reduce val loss across ranks so all GPUs agree ───────────────────
        vl_tensor = torch.tensor(vl, device=device)
        dist.all_reduce(vl_tensor, op=dist.ReduceOp.AVG)
        vl = vl_tensor.item()

        elapsed = time.time() - t0

        # ── Log (rank-0 only) ─────────────────────────────────────────────────
        is_best = vl < best_val_loss
        if is_best:
            best_val_loss = vl
            no_improve    = 0
        else:
            no_improve   += 1

        for k, v in zip(
            ["tl","vl","tpa","vpa","toa","voa","vaa","degree","power_state","lr"],
            [tl, vl, tpa, vpa, toa, voa, vaa,
             throttle.prefetch_degree, throttle.state, current_lr]
        ):
            history[k].append(v if isinstance(v, str) else round(float(v), 6))

        log(rank,
            f"{epoch:>4}  {tl:>9.4f}  {vl:>9.4f}  "
            f"{vpa*100:>10.2f}%  {vaa*100:>10.2f}%  "
            f"{throttle.prefetch_degree:>6}x  {throttle.state:>5}  "
            f"{current_lr:>9.2e}  {elapsed:>5.1f}s"
        )

        # ── Save checkpoint ───────────────────────────────────────────────────
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss,
                        history, CFG["ckpt_dir"], is_best=is_best, rank=rank)

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve >= CFG["early_stop_pat"]:
            log(rank, f"\n  [EARLY STOP] No improvement for {no_improve} epochs — stopping.")
            break

        dist.barrier()

    # ── Training complete ─────────────────────────────────────────────────────
    log(rank, "\n" + "─"*80)
    log(rank, "Training complete!")
    if rank == 0:
        throttle.summary()

    # ── Save final .pt weights ────────────────────────────────────────────────
    save_final_pt(model, CFG["ckpt_dir"], rank=rank)

    # ── Save training history as JSON ─────────────────────────────────────────
    if rank == 0:
        hist_path = os.path.join(CFG["ckpt_dir"], "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        log(rank, f"  [LOG] History → {hist_path}")

    # ── Final test evaluation (rank-0 only) ───────────────────────────────────
    if rank == 0:
        log(rank, "\nRunning final test evaluation ...")
        # Load best model weights
        best_ckpt = os.path.join(CFG["ckpt_dir"], "mambaedge_best.pth")
        if os.path.exists(best_ckpt):
            best_state = torch.load(best_ckpt, map_location=device)
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.load_state_dict(best_state["model_state"])

        raw_model = model.module if hasattr(model, "module") else model
        raw_model.eval()

        page_correct = offset_correct = addr_correct = topk_correct = total = 0
        with torch.no_grad():
            for Xpb, Xob, ypb, yob in test_dl:
                Xpb, Xob = Xpb.to(device), Xob.to(device)
                ypb, yob = ypb.to(device), yob.to(device)

                pl, ol      = raw_model(Xpb, Xob)
                pred_page   = pl.argmax(-1)
                pred_offset = ol.argmax(-1)

                page_correct   += (pred_page   == ypb).sum().item()
                offset_correct += (pred_offset == yob).sum().item()

                pred_addr = (pred_page.long() << 12) | (pred_offset.long() << 6)
                true_addr = (ypb.long()       << 12) | (yob.long()         << 6)
                addr_correct += (pred_addr == true_addr).sum().item()

                top3 = pl.topk(3, dim=-1).indices
                topk_correct += (top3 == ypb.unsqueeze(-1)).any(-1).sum().item()
                total        += ypb.size(0)

        page_acc   = page_correct   / max(1, total)
        offset_acc = offset_correct / max(1, total)
        addr_acc   = addr_correct   / max(1, total)
        topk_acc   = topk_correct   / max(1, total)

        print("="*55)
        print("   MambaEdge — Final Test Results (best model)")
        print("="*55)
        print(f"  Page  accuracy   (exact) : {page_acc*100:6.2f}%")
        print(f"  Page  coverage   (top-3) : {topk_acc*100:6.2f}%")
        print(f"  Offset accuracy  (exact) : {offset_acc*100:6.2f}%")
        print(f"  ─────────────────────────────────────")
        print(f"  Address accuracy         : {addr_acc*100:6.2f}%")
        print(f"  (page<<12 | offset<<6 — both must match)")
        print("="*55)
        print(f"\nCheckpoint files saved in: {os.path.abspath(CFG['ckpt_dir'])}/")
        print(f"  mambaedge_best.pth   ← best checkpoint (resume + weights)")
        print(f"  mambaedge_latest.pth ← symlink to last epoch (for resuming)")
        print(f"  mambaedge_final.pt   ← final weights only (inference)")
        print(f"  training_history.json")

    cleanup_ddp()


if __name__ == "__main__":
    main()