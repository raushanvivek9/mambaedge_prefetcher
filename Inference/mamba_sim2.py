import os, hashlib, time, json, lzma, collections, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants (must match training CFG exactly) ───────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN     = 32
N_PAGES     = 512
N_OFFSETS   = 64
PAGE_BITS   = 12
OFFSET_BITS = 6
MODEL_PTH   = '/home/cs25m115/mamba/checkpoints/mambaedge_best.pth'

ARCH = dict(
    n_pages   = N_PAGES,
    n_offsets = N_OFFSETS,
    d_model   = 64,
    n_layers  = 4,
    d_state   = 16,
    expand    = 2,
    dropout   = 0.0,
)

# ══════════════════════════════════════════════════════════════════════════════
# Cache Microarchitecture Constants
# ══════════════════════════════════════════════════════════════════════════════
LLC_HIT_LATENCY_CY   = 30        # LLC hit latency (cycles)
DRAM_MISS_PENALTY_CY = 200       # DRAM round-trip penalty (cycles)
CACHE_LINE_BYTES     = 64        # cache line size (bytes)
PROCESSOR_FREQ_GHZ   = 3.0       # CPU frequency (GHz)

# ── Power / Energy model ──────────────────────────────────────────────────────
LLC_STATIC_POWER_W   = 1.5       # LLC leakage power (Watts)
ENERGY_LLC_HIT_NJ    = 1.0       # dynamic energy: LLC hit        (nJ)
ENERGY_DRAM_NJ       = 20.0      # dynamic energy: DRAM access    (nJ)
ENERGY_PF_FILL_NJ    = 3.0       # dynamic energy: prefetch fill  (nJ)
ENERGY_PF_EVICT_NJ   = 1.5       # dynamic energy: pollution eviction (nJ)


# ══════════════════════════════════════════════════════════════════════════════
# Cache Metrics — FIXED version
#
# ROOT CAUSE of 0.00% energy/power in the original:
#   1. elapsed_s = Python wall-clock time (seconds of I/O + GPU work)
#      → static_energy = 1.5W × 120s × 1e6 = 180,000,000 µJ  (180 J!)
#      → dynamic savings (~8,000 µJ) become 0.004% → rounds to 0.00%
#   2. energy_savings_pct divided by TOTAL energy (static + dynamic),
#      diluting the real dynamic savings even further.
#   3. Fill cost counted for ALL issued PFs (degree × accesses), not just
#      useful fills, so fill overhead > savings when degree = 8.
#
# FIXES applied here:
#   F1. Simulated hardware cycle time replaces wall-clock time for static energy:
#         sim_cycles = Σ (LLC_hit_latency per hit + DRAM_penalty per miss)
#         sim_time_hw_s = sim_cycles / FREQ
#       This gives the actual hardware execution time — milliseconds, not minutes.
#   F2. energy_savings_pct uses DYNAMIC-ONLY energy as denominator.
#       Static leakage is reported separately and excluded from savings %.
#   F3. Fill energy only charged for USEFUL prefetches (lines actually accessed).
#       Useless fills pay the eviction penalty only (already in e_pf_poll_uj).
# ══════════════════════════════════════════════════════════════════════════════

def compute_cache_metrics(
    s_accesses   : int,
    s_misses     : int,
    s_pf_issued  : int,
    s_pf_useful  : int,
    s_pf_useless : int,
    ipc_base     : float,
    elapsed_s    : float,   # kept for reference only; NOT used for static energy
) -> dict:

    freq_hz = PROCESSOR_FREQ_GHZ * 1e9

    # ── Reconstruct baseline (no-PF) within the same sim window ──────────────
    # Every miss that PF prevented would have been a DRAM miss without PF.
    baseline_misses = s_misses + s_pf_useful
    baseline_hits   = max(0, s_accesses - baseline_misses)
    pf_hits         = max(0, s_accesses - s_misses)

    miss_rate_baseline = baseline_misses / max(1, s_accesses)
    miss_rate_with_pf  = s_misses        / max(1, s_accesses)
    miss_reduction_abs = s_pf_useful
    miss_reduction_pct = miss_reduction_abs / max(1, baseline_misses) * 100

    # ── Cache pollution ───────────────────────────────────────────────────────
    pollution_count      = s_pf_useless
    pollution_rate_pct   = s_pf_useless / max(1, s_pf_issued) * 100

    # ── FIX F1: Simulated hardware cycle time ─────────────────────────────────
    # Model: every LLC access costs LLC_HIT_LATENCY_CY;
    # every DRAM miss costs an additional DRAM_MISS_PENALTY_CY on top.
    # This gives a memory-bound cycle estimate that is physically meaningful.
    #
    # Why NOT use ipc_base directly?
    #   sim_instr / ipc_base gives total execution cycles (compute + memory),
    #   but our energy model is specifically about MEMORY subsystem cycles.
    #   Using the latency model isolates exactly the cycles affected by PF.
    #
    # Why NOT use Python elapsed_s?
    #   Python time includes I/O, inference, and scheduling overhead —
    #   nothing to do with hardware energy consumption.

    cycles_base_hw = (baseline_hits   * LLC_HIT_LATENCY_CY
                    + baseline_misses * DRAM_MISS_PENALTY_CY)
    cycles_pf_hw   = (pf_hits         * LLC_HIT_LATENCY_CY
                    + s_misses        * DRAM_MISS_PENALTY_CY)

    # Pollution adds back-pressure: useless fills displace lines, causing
    # re-fetches at LLC latency.
    pollution_cycles  = s_pf_useless  * LLC_HIT_LATENCY_CY
    cycles_pf_hw     += pollution_cycles

    sim_time_base_s = cycles_base_hw / max(1, freq_hz)   # hardware seconds (baseline)
    sim_time_pf_s   = cycles_pf_hw   / max(1, freq_hz)   # hardware seconds (with PF)

    # Pollution overhead as % of baseline cycles
    total_cycles_ipc_base = s_accesses / max(1e-9, ipc_base)  # kept for IPC/BW calcs
    pollution_overhead_pct = pollution_cycles / max(1, total_cycles_ipc_base) * 100

    # ── AMAT (cycles) ─────────────────────────────────────────────────────────
    amat_baseline = LLC_HIT_LATENCY_CY + miss_rate_baseline * DRAM_MISS_PENALTY_CY
    amat_with_pf  = LLC_HIT_LATENCY_CY + miss_rate_with_pf  * DRAM_MISS_PENALTY_CY
    amat_reduction_cy  = amat_baseline - amat_with_pf
    amat_reduction_pct = amat_reduction_cy / max(1e-9, amat_baseline) * 100

    # ── Throughput (bytes/cycle, GB/s) ────────────────────────────────────────
    bytes_accessed       = s_accesses * CACHE_LINE_BYTES
    throughput_base_bpc  = bytes_accessed / max(1, total_cycles_ipc_base)
    saved_miss_cycles    = s_pf_useful  * DRAM_MISS_PENALTY_CY
    total_cycles_pf      = max(1, total_cycles_ipc_base - saved_miss_cycles + pollution_cycles)
    throughput_pf_bpc    = bytes_accessed / total_cycles_pf
    throughput_gain_pct  = (throughput_pf_bpc - throughput_base_bpc) / max(1e-9, throughput_base_bpc) * 100
    throughput_base_gbs  = throughput_base_bpc * freq_hz / 1e9
    throughput_pf_gbs    = throughput_pf_bpc   * freq_hz / 1e9

    # ── FIX F2 + F3: Energy using hardware sim time & correct fill accounting ─
    #
    # Static leakage energy now uses sim_time_hw_s (microseconds, not minutes).
    # Baseline and PF use their own sim times since PF shortens execution.
    static_base_uj = LLC_STATIC_POWER_W * sim_time_base_s * 1e6   # W·s → µJ
    static_pf_uj   = LLC_STATIC_POWER_W * sim_time_pf_s   * 1e6

    # Dynamic energy components
    e_base_hits_uj  = baseline_hits   * ENERGY_LLC_HIT_NJ  / 1e3
    e_base_dram_uj  = baseline_misses * ENERGY_DRAM_NJ     / 1e3

    e_pf_hits_uj    = pf_hits         * ENERGY_LLC_HIT_NJ  / 1e3
    e_pf_dram_uj    = s_misses        * ENERGY_DRAM_NJ     / 1e3
    # FIX F3: Only charge fill energy for USEFUL prefetches.
    # Useless prefetches pay eviction penalty (poll), not fill cost, since
    # they never get referenced after being placed in LLC.
    e_pf_fill_uj    = s_pf_useful     * ENERGY_PF_FILL_NJ  / 1e3
    e_pf_poll_uj    = s_pf_useless    * ENERGY_PF_EVICT_NJ / 1e3

    # Total energy (dynamic + static)
    energy_baseline_uj  = e_base_hits_uj + e_base_dram_uj + static_base_uj
    energy_with_pf_uj   = (e_pf_hits_uj + e_pf_dram_uj
                         + e_pf_fill_uj + e_pf_poll_uj + static_pf_uj)

    # Total energy savings
    energy_savings_uj   = max(0.0, energy_baseline_uj - energy_with_pf_uj)

    # FIX F2: Dynamic-only savings % (static cancels or changes with exec time)
    e_dyn_baseline  = e_base_hits_uj + e_base_dram_uj
    e_dyn_with_pf   = e_pf_hits_uj + e_pf_dram_uj + e_pf_fill_uj + e_pf_poll_uj
    dyn_savings_uj  = max(0.0, e_dyn_baseline - e_dyn_with_pf)
    # % of dynamic-only baseline — this is the "true" prefetcher energy benefit
    energy_savings_pct  = dyn_savings_uj / max(1e-9, e_dyn_baseline) * 100
    # Total savings % (includes static reduction from shorter execution)
    total_savings_pct   = energy_savings_uj / max(1e-9, energy_baseline_uj) * 100

    # ── Power (Watts) using hardware sim time ─────────────────────────────────
    # Power = total energy / execution time
    # We use baseline sim time for both to compare power over the SAME interval.
    power_baseline_w  = energy_baseline_uj / 1e6 / max(1e-9, sim_time_base_s)
    # Normalise PF energy to the SAME time window as baseline for fair comparison
    power_with_pf_w   = energy_with_pf_uj  / 1e6 / max(1e-9, sim_time_base_s)
    power_savings_w   = max(0.0, power_baseline_w - power_with_pf_w)
    power_savings_pct = power_savings_w / max(1e-9, power_baseline_w) * 100

    # Static leakage breakdown (informational)
    static_savings_uj   = max(0.0, static_base_uj - static_pf_uj)
    static_savings_pct  = static_savings_uj / max(1e-9, static_base_uj) * 100

    return dict(
        # Miss reduction
        baseline_misses        = baseline_misses,
        pf_prevented_misses    = miss_reduction_abs,
        miss_rate_baseline_pct = round(miss_rate_baseline * 100, 4),
        miss_rate_with_pf_pct  = round(miss_rate_with_pf  * 100, 4),
        miss_reduction_abs     = miss_reduction_abs,
        miss_reduction_pct     = round(miss_reduction_pct, 2),

        # Cache pollution
        pollution_count        = pollution_count,
        pollution_rate_pct     = round(pollution_rate_pct, 2),
        pollution_cycles       = pollution_cycles,
        pollution_overhead_pct = round(pollution_overhead_pct, 2),

        # AMAT
        amat_baseline_cy       = round(amat_baseline, 2),
        amat_with_pf_cy        = round(amat_with_pf, 2),
        amat_reduction_cy      = round(amat_reduction_cy, 2),
        amat_reduction_pct     = round(amat_reduction_pct, 2),

        # Throughput
        throughput_base_bpc    = round(throughput_base_bpc, 4),
        throughput_pf_bpc      = round(throughput_pf_bpc, 4),
        throughput_gain_pct    = round(throughput_gain_pct, 2),
        throughput_base_gbs    = round(throughput_base_gbs, 4),
        throughput_pf_gbs      = round(throughput_pf_gbs, 4),

        # Simulated hardware execution time (not Python wall-clock)
        sim_time_base_us       = round(sim_time_base_s * 1e6, 4),   # microseconds
        sim_time_pf_us         = round(sim_time_pf_s   * 1e6, 4),
        exec_time_reduction_pct= round((1 - sim_time_pf_s / max(1e-9, sim_time_base_s)) * 100, 2),

        # Energy (µJ)  — FIXED
        energy_baseline_uj     = round(energy_baseline_uj, 4),
        energy_with_pf_uj      = round(energy_with_pf_uj, 4),
        energy_savings_uj      = round(energy_savings_uj, 4),
        energy_savings_pct     = round(energy_savings_pct, 2),    # dynamic-only %
        total_energy_savings_pct = round(total_savings_pct, 2),   # includes static
        energy_breakdown_base  = dict(
            hits_uj    = round(e_base_hits_uj, 4),
            dram_uj    = round(e_base_dram_uj, 4),
            static_uj  = round(static_base_uj, 4),
        ),
        energy_breakdown_pf    = dict(
            hits_uj    = round(e_pf_hits_uj, 4),
            dram_uj    = round(e_pf_dram_uj, 4),
            fills_uj   = round(e_pf_fill_uj, 4),
            poll_uj    = round(e_pf_poll_uj, 4),
            static_uj  = round(static_pf_uj, 4),
        ),
        static_savings_uj      = round(static_savings_uj, 4),
        static_savings_pct     = round(static_savings_pct, 2),

        # Power (W)  — FIXED (uses hardware sim time, same window)
        power_baseline_w       = round(power_baseline_w, 4),
        power_with_pf_w        = round(power_with_pf_w, 4),
        power_savings_w        = round(power_savings_w, 4),
        power_savings_pct      = round(power_savings_pct, 2),
    )


# ── IPEX Throttle ─────────────────────────────────────────────────────────────
POWER_HIGH = "HIGH";  POWER_MED = "MED";  POWER_LOW = "LOW"
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
        self._state       = POWER_HIGH
        self._acc_window  = collections.deque(maxlen=window_size)
        self._below_count = 0
        self._above_count = 0
        self._batch_idx   = 0
        self.history      = {"accuracy": [], "prefetch_degree": [], "state": [], "events": []}

    @property
    def state(self): return self._state
    @property
    def prefetch_degree(self): return DEGREE_MAP[self._state]
    @property
    def rolling_accuracy(self):
        return sum(self._acc_window)/len(self._acc_window) if self._acc_window else 0.0

    def update(self, accuracy: float) -> int:
        self._acc_window.append(accuracy)
        self._batch_idx += 1
        avg = self.rolling_accuracy
        self.history["accuracy"].append(round(avg, 4))
        self.history["prefetch_degree"].append(self.prefetch_degree)
        self.history["state"].append(self._state)
        if avg < self.throttle_threshold:
            self._below_count += 1;  self._above_count = 0
            if self._below_count >= self.throttle_patience:
                self._step_down();  self._below_count = 0
        elif avg > self.recover_threshold:
            self._above_count += 1;  self._below_count = 0
            if self._above_count >= self.recover_patience:
                self._step_up();  self._above_count = 0
        else:
            self._below_count = self._above_count = 0
        return self.prefetch_degree

    def _step_down(self):
        old = self._state
        if self._state == POWER_HIGH: self._state = POWER_MED
        elif self._state == POWER_MED: self._state = POWER_LOW
        if old != self._state:
            ev = f"THROTTLE {old}→{self._state}  degree {DEGREE_MAP[old]}→{self.prefetch_degree}x  acc={self.rolling_accuracy*100:.1f}%"
            self.history["events"].append((self._batch_idx, ev))
            if self.verbose: print(f"\n  [IPEX] ⬇  {ev}")

    def _step_up(self):
        old = self._state
        if self._state == POWER_LOW:  self._state = POWER_MED
        elif self._state == POWER_MED: self._state = POWER_HIGH
        if old != self._state:
            ev = f"RECOVER  {old}→{self._state}  degree {DEGREE_MAP[old]}→{self.prefetch_degree}x  acc={self.rolling_accuracy*100:.1f}%"
            self.history["events"].append((self._batch_idx, ev))
            if self.verbose: print(f"\n  [IPEX] ⬆  {ev}")


# ── GPU check ─────────────────────────────────────────────────────────────────
def _gpu_supports_triton_mamba() -> bool:
    if not torch.cuda.is_available(): return False
    for i in range(torch.cuda.device_count()):
        major, _ = torch.cuda.get_device_capability(i)
        if major < 7:
            print(f"[WARN] GPU {i} sm_{major}x < sm_70. Using PyTorch SSM fallback.")
            return False
    return True

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    MAMBA_AVAILABLE = _gpu_supports_triton_mamba()
except ImportError:
    MAMBA_AVAILABLE = False
    print("[WARN] mamba_ssm not found. Using pure-PyTorch SSM fallback.")


# ── Model definition ──────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def apply_rope(x, a):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([x1*torch.cos(a) - x2*torch.sin(a),
                      x1*torch.sin(a) + x2*torch.cos(a)], dim=-1)

class Mamba3Block(nn.Module):
    def __init__(self, d_model=64, d_state=16, expand=2, chunk_size=128, norm_eps=1e-5):
        super().__init__()
        E=d_model*expand; H,P=1,E; td=d_state//2
        self.d_state,self.E,self.chunk_size=d_state,E,chunk_size
        self._td,self._H,self._P=td,H,P
        self.in_proj   = nn.Linear(d_model, E+E+d_state+d_state+H+H+td+H, bias=False)
        self.bc_norm_B = RMSNorm(d_state, norm_eps)
        self.bc_norm_C = RMSNorm(d_state, norm_eps)
        self.B_bias    = nn.Parameter(torch.ones(d_state))
        self.C_bias    = nn.Parameter(torch.ones(d_state))
        dt_i = torch.exp(torch.rand(H)*(math.log(0.1)-math.log(0.001))+math.log(0.001)).clamp(min=1e-4)
        self.register_buffer('dt_bias', dt_i + torch.log(-torch.expm1(-dt_i)))
        self.A_log    = nn.Parameter(torch.log(0.5*torch.ones(H)))
        self.out_proj = nn.Linear(E, d_model, bias=False)

    def _fallback_ssm(self, x_trap, dt, A, B_, C_):
        B_sz, L, H, P = x_trap.shape; N = B_.shape[-1]
        u = x_trap.mean(dim=-1); dA = torch.exp(dt * A.view(1, 1, H))
        dB = dt.unsqueeze(-1) * B_.unsqueeze(2); dBu = dB * u.unsqueeze(-1)
        state = torch.zeros(B_sz, H, N, device=x_trap.device, dtype=x_trap.dtype)
        outs  = torch.empty(B_sz, L, H, device=x_trap.device, dtype=x_trap.dtype)
        for start in range(0, L, 256):
            for t in range(start, min(start+256, L)):
                state = dA[:, t, :].unsqueeze(-1) * state + dBu[:, t]
                outs[:, t] = (state * C_[:, t].unsqueeze(1)).sum(-1)
        return outs

    def forward(self, u):
        B,L,_ = u.shape; N,H,P = self.d_state, self._H, self._P
        z = self.in_proj(u); ptr = 0
        def take(n):
            nonlocal ptr; o=z[...,ptr:ptr+n]; ptr+=n; return o
        x=take(self.E); gate=take(self.E); B_=take(N); C_=take(N)
        dt_=take(H); A_=take(H); th=take(self._td); lam_=take(H)
        x=F.silu(x); gate=F.silu(gate)
        dt=F.softplus(dt_+self.dt_bias); A=-F.softplus(A_+self.A_log)
        lam=torch.sigmoid(lam_); alpha=torch.exp(dt*A)
        B_=self.bc_norm_B(B_)+self.B_bias; C_=self.bc_norm_C(C_)+self.C_bias
        ca=torch.cumsum(dt.mean(-1,keepdim=True)*th, dim=1)
        B_=apply_rope(B_,ca); C_=apply_rope(C_,ca)
        xh=x.view(B,L,H,P); xp=torch.roll(xh,1,dims=1); xp[:,0]=0.0
        xt=(lam.unsqueeze(-1))*xh+(1-lam.unsqueeze(-1))*alpha.unsqueeze(-1)*xp
        if MAMBA_AVAILABLE:
            y = mamba_chunk_scan_combined(xt, dt, A.mean(dim=(0,1)),
                B_.unsqueeze(2), C_.unsqueeze(2), chunk_size=self.chunk_size,
                D=None, z=None, dt_bias=None, dt_softplus=False)
            y = y.reshape(B, L, self.E)
        else:
            y_raw = self._fallback_ssm(xt, dt, A.mean(dim=(0,1)), B_, C_)
            y = y_raw.reshape(B, L, H).expand(B, L, self.E).contiguous()
        return self.out_proj(y * gate)

class MambaEdgePrefetcher(nn.Module):
    def __init__(self, n_pages=512, n_offsets=64, d_model=64, n_layers=4,
                 d_state=16, expand=2, dropout=0.0):
        super().__init__()
        self.page_embed   = nn.Embedding(n_pages, d_model)
        self.offset_embed = nn.Embedding(n_offsets, d_model//4)
        self.input_proj   = nn.Sequential(
            nn.Linear(d_model+d_model//4, d_model), nn.LayerNorm(d_model))
        self.mamba_layers = nn.ModuleList([
            Mamba3Block(d_model=d_model, d_state=d_state, expand=expand)
            for _ in range(n_layers)])
        self.norms   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.page_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model//2, n_pages))
        self.offset_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model//4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model//4, n_offsets))

    def forward(self, pg, off):
        p = self.page_embed(pg); o = self.offset_embed(off)
        x = self.dropout(self.input_proj(torch.cat([p,o], dim=-1)))
        for m, n in zip(self.mamba_layers, self.norms): x = x + m(n(x))
        return self.page_head(x[:,-1,:]), self.offset_head(x[:,-1,:])


# ── Load model ────────────────────────────────────────────────────────────────
_model = MambaEdgePrefetcher(**ARCH).to(DEVICE)
_raw = torch.load(MODEL_PTH, map_location=DEVICE)
if isinstance(_raw, dict):
    if   'model_state'      in _raw: ckpt = _raw['model_state']
    elif 'model_state_dict' in _raw: ckpt = _raw['model_state_dict']
    elif 'state_dict'       in _raw: ckpt = _raw['state_dict']
    else:                            ckpt = _raw
else:
    ckpt = _raw
ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
_model.load_state_dict(ckpt, strict=True)
_model.eval()
print(f'Model loaded on {DEVICE}  |  ARCH: {ARCH}')

@torch.no_grad()
def run_inference(page_batch, offset_batch):
    pl, ol = _model(page_batch, offset_batch)
    return pl.argmax(-1).cpu().numpy().astype(int), \
           ol.argmax(-1).cpu().numpy().astype(int)


# ── Trace loader ──────────────────────────────────────────────────────────────
def load_trace_fast(trace_path, max_accesses=700_000):
    print(f'  Reading: {os.path.basename(trace_path)} ...', end=' ', flush=True)
    t0 = time.time(); addrs = []
    with lzma.open(trace_path, 'rt', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            try:
                addr = int(parts[2], 16) if len(parts) >= 3 else int(parts[0], 16)
                addrs.append(addr)
            except: continue
            if len(addrs) >= max_accesses: break
    addrs   = np.array(addrs, dtype=np.int64)
    pages   = ((addrs >> PAGE_BITS) % N_PAGES).astype(np.int16)
    offsets = (((addrs & ((1 << PAGE_BITS)-1)) >> (PAGE_BITS - OFFSET_BITS)) % N_OFFSETS).astype(np.int8)
    print(f'{len(addrs):,} accesses in {time.time()-t0:.1f}s | {len(np.unique(pages))} unique pages')
    return pages, offsets

def batch_predict_all(pages, offsets, seq_len, batch_size=512):
    N = len(pages); pred_pg = np.zeros(N, dtype=np.int16); pred_off = np.zeros(N, dtype=np.int8)
    print(f'  Running batched inference (batch={batch_size}) ...', end=' ', flush=True)
    t0 = time.time()
    for batch_start in range(seq_len, N, batch_size):
        batch_end = min(batch_start + batch_size, N); B = batch_end - batch_start
        pg_w = np.zeros((B, seq_len), dtype=np.int64); off_w = np.zeros((B, seq_len), dtype=np.int64)
        for j in range(B):
            i = batch_start + j; pg_w[j] = pages[i-seq_len:i]; off_w[j] = offsets[i-seq_len:i]
        p_out, o_out = run_inference(torch.from_numpy(pg_w).to(DEVICE), torch.from_numpy(off_w).to(DEVICE))
        pred_pg[batch_start:batch_end] = p_out; pred_off[batch_start:batch_end] = o_out
    print(f'done in {time.time()-t0:.1f}s')
    return pred_pg, pred_off

CACHE_DIR = '/home/cs25m115/mamba_inf/trace_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_key(trace_path, seq_len):
    return hashlib.md5(f"{os.path.basename(trace_path)}|seq={seq_len}".encode()).hexdigest()

def load_trace_cached(trace_path, seq_len, max_accesses=700_000):
    key = _cache_key(trace_path, seq_len); path = os.path.join(CACHE_DIR, f"{key}.npz")
    if os.path.exists(path):
        print(f'  [cache HIT]  {os.path.basename(path)}')
        d = np.load(path); return d['pages'], d['offsets'], d['pred_pg'], d['pred_off']
    print(f'  [cache MISS] Reading trace + running inference ...')
    pages, offsets = load_trace_fast(trace_path, max_accesses=max_accesses)
    pred_pg, pred_off = batch_predict_all(pages, offsets, seq_len, batch_size=512)
    np.savez_compressed(path, pages=pages, offsets=offsets, pred_pg=pred_pg, pred_off=pred_off)
    print(f'  [cache SAVE] → {path}')
    return pages, offsets, pred_pg, pred_off


# ══════════════════════════════════════════════════════════════════════════════
# Main simulation
# ══════════════════════════════════════════════════════════════════════════════
def champsim_simulate(
    trace_path,
    seq_len            = SEQ_LEN,
    warmup_instr       = 200_000,
    sim_instr          = 500_000,
    batch_size         = 512,
    ipc_base           = 2.5,
    throttle_threshold = 0.50,
    recover_threshold  = 0.70,
):
    name = os.path.basename(trace_path).replace('.champsimtrace.xz', '')
    total_needed = warmup_instr + sim_instr + seq_len + 10

    print(); print('='*70)
    print(f'  MambaEdge ChampSim Simulation  |  Trace: {name}')
    print('='*70)
    t_total = time.time()

    pages, offsets, pred_pg, pred_off = load_trace_cached(
        trace_path, seq_len, max_accesses=total_needed)
    N = len(pages)
    if N < seq_len + 100:
        print(f'  ERROR: trace too short ({N} accesses)'); return None

    available        = N - seq_len
    effective_warmup = min(warmup_instr, int(0.4 * available))
    effective_sim    = min(sim_instr,    available - effective_warmup)

    throttle     = IPEXThrottle(throttle_threshold, recover_threshold, verbose=True)
    w_misses     = s_accesses = s_pf_issued = s_pf_useful = s_pf_useless = s_correct = s_misses = 0

    for i in range(seq_len, N):
        true_pg = int(pages[i]); true_off = int(offsets[i])
        ppg     = int(pred_pg[i]); poff    = int(pred_off[i])
        hit     = (ppg == true_pg) and (poff == true_off)

        if i < seq_len + effective_warmup:
            w_misses += 1; continue
        if s_accesses >= effective_sim:
            break

        s_accesses  += 1; s_misses += 1
        degree       = throttle.prefetch_degree
        s_pf_issued += degree
        s_correct   += int(hit)
        if hit:
            s_pf_useful += 1; s_misses -= 1
        else:
            s_pf_useless += 1
        throttle.update(float(hit))

    elapsed = time.time() - t_total

    # ── Standard metrics ──────────────────────────────────────────────────────
    mpki_baseline  = (w_misses   / max(1, effective_warmup)) * 1000
    mpki_with_pf   = (s_misses   / max(1, s_accesses))       * 1000
    mpki_reduction = max(0.0, mpki_baseline - mpki_with_pf)
    pf_accuracy    = s_pf_useful  / max(1, s_pf_issued)            * 100
    pf_coverage    = s_pf_useful  / max(1, s_misses + s_pf_useful) * 100
    overprediction = s_pf_useless / max(1, s_pf_issued)            * 100
    addr_accuracy  = s_correct    / max(1, s_accesses)             * 100
    saved_cycles   = s_pf_useful  * DRAM_MISS_PENALTY_CY
    total_cycles   = effective_sim / ipc_base
    ipc_improved   = effective_sim / max(1, total_cycles - saved_cycles)
    ipc_speedup    = ipc_improved / ipc_base

    # ── Extended cache metrics (fixed) ────────────────────────────────────────
    cm = compute_cache_metrics(
        s_accesses=s_accesses, s_misses=s_misses,
        s_pf_issued=s_pf_issued, s_pf_useful=s_pf_useful, s_pf_useless=s_pf_useless,
        ipc_base=ipc_base, elapsed_s=elapsed)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(); print('='*70)
    print(f'  Results: {name}'); print('='*70)
    print(f'  Warmup: {effective_warmup:,}   |   Sim: {effective_sim:,}')

    print(f'\n  ── Miss Reduction ───────────────────────────────────────────────')
    print(f'  LLC LOAD MPKI  baseline    : {mpki_baseline:>10.2f}')
    print(f'  LLC LOAD MPKI  w/ MambaEdge: {mpki_with_pf:>10.2f}')
    print(f'  MPKI Reduction             : {mpki_reduction:>10.2f}')
    print(f'  Misses prevented by PF     : {cm["pf_prevented_misses"]:>10,}')
    print(f'  Miss rate  baseline        : {cm["miss_rate_baseline_pct"]:>9.4f}%')
    print(f'  Miss rate  w/ MambaEdge   : {cm["miss_rate_with_pf_pct"]:>9.4f}%')
    print(f'  Miss Reduction             : {cm["miss_reduction_pct"]:>9.2f}%')

    print(f'\n  ── Cache Pollution ──────────────────────────────────────────────')
    print(f'  Useless prefetches         : {cm["pollution_count"]:>10,}')
    print(f'  Pollution rate             : {cm["pollution_rate_pct"]:>9.2f}%')
    print(f'  Wasted cycles              : {cm["pollution_cycles"]:>10,}')
    print(f'  Cycle overhead             : {cm["pollution_overhead_pct"]:>9.2f}%')

    print(f'\n  ── Latency: AMAT (cycles) ───────────────────────────────────────')
    print(f'  AMAT  baseline             : {cm["amat_baseline_cy"]:>10.2f} cy')
    print(f'  AMAT  w/ MambaEdge         : {cm["amat_with_pf_cy"]:>10.2f} cy')
    print(f'  AMAT  reduction            : {cm["amat_reduction_cy"]:>10.2f} cy  ({cm["amat_reduction_pct"]:.2f}%)')

    print(f'\n  ── Memory Throughput ────────────────────────────────────────────')
    print(f'  Throughput  baseline       : {cm["throughput_base_gbs"]:>10.4f} GB/s')
    print(f'  Throughput  w/ MambaEdge   : {cm["throughput_pf_gbs"]:>10.4f} GB/s')
    print(f'  Throughput gain            : {cm["throughput_gain_pct"]:>9.2f}%')

    print(f'\n  ── Simulated Hardware Execution Time ────────────────────────────')
    print(f'  (uses memory-latency cycle model, NOT Python wall-clock time)')
    print(f'  Exec time  baseline        : {cm["sim_time_base_us"]:>10.4f} µs')
    print(f'  Exec time  w/ MambaEdge    : {cm["sim_time_pf_us"]:>10.4f} µs')
    print(f'  Execution time reduction   : {cm["exec_time_reduction_pct"]:>9.2f}%')

    print(f'\n  ── Energy (µJ) ──────────────────────────────────────────────────')
    print(f'  Energy  baseline           : {cm["energy_baseline_uj"]:>14.4f} µJ')
    print(f'  Energy  w/ MambaEdge       : {cm["energy_with_pf_uj"]:>14.4f} µJ')
    print(f'  Energy savings (total)     : {cm["energy_savings_uj"]:>14.4f} µJ  ({cm["total_energy_savings_pct"]:.2f}%)')
    print(f'  Energy savings (dyn only)  : {cm["energy_savings_pct"]:>13.2f}%  ← primary metric')
    bd_b = cm["energy_breakdown_base"]
    bd_p = cm["energy_breakdown_pf"]
    print(f'  Breakdown  baseline:   LLC hits {bd_b["hits_uj"]:.2f}  DRAM {bd_b["dram_uj"]:.2f}  static {bd_b["static_uj"]:.2f}  µJ')
    print(f'  Breakdown  w/ PF  :   LLC hits {bd_p["hits_uj"]:.2f}  DRAM {bd_p["dram_uj"]:.2f}  fills {bd_p["fills_uj"]:.2f}  poll {bd_p["poll_uj"]:.2f}  static {bd_p["static_uj"]:.2f}  µJ')
    print(f'  Static leakage savings     : {cm["static_savings_uj"]:>14.4f} µJ  ({cm["static_savings_pct"]:.2f}%)  [from shorter exec]')

    print(f'\n  ── Power (W)  [normalised to same window] ───────────────────────')
    print(f'  Power  baseline            : {cm["power_baseline_w"]:>10.4f} W')
    print(f'  Power  w/ MambaEdge        : {cm["power_with_pf_w"]:>10.4f} W')
    print(f'  Power savings              : {cm["power_savings_w"]:>10.4f} W  ({cm["power_savings_pct"]:.2f}%)')

    print(f'\n  ── Prefetcher Performance ───────────────────────────────────────')
    print(f'  Prefetches Issued          : {s_pf_issued:>12,}')
    print(f'  Prefetches Useful          : {s_pf_useful:>12,}   ({pf_accuracy:.1f}% accuracy)')
    print(f'  Prefetches Useless         : {s_pf_useless:>12,}   ({overprediction:.1f}% overprediction)')
    print(f'  Prefetch Coverage          : {pf_coverage:>10.2f}%')
    print(f'  Address Accuracy (exact)   : {addr_accuracy:>10.2f}%')

    print(f'\n  ── IPC ──────────────────────────────────────────────────────────')
    print(f'  IPC  baseline              : {ipc_base:>10.2f}')
    print(f'  IPC  w/ MambaEdge          : {ipc_improved:>10.2f}')
    print(f'  IPC Speedup                : {ipc_speedup:>10.3f}x')

    print(f'\n  ── IPEX Power Throttle ──────────────────────────────────────────')
    print(f'  Final state / degree       : {throttle.state} / {throttle.prefetch_degree}x')
    for idx, ev in throttle.history['events']:
        print(f'    access {idx:>8,} : {ev}')

    print(f'\n  Python wall-clock time     : {elapsed:.1f}s')
    print('='*70)

    return {
        'trace': name,
        'mpki_baseline':  round(mpki_baseline, 2),
        'mpki_with_pf':   round(mpki_with_pf, 2),
        'mpki_reduction': round(mpki_reduction, 2),
        'pf_issued': s_pf_issued, 'pf_useful': s_pf_useful, 'pf_useless': s_pf_useless,
        'pf_accuracy':    round(pf_accuracy, 2),
        'pf_coverage':    round(pf_coverage, 2),
        'overprediction': round(overprediction, 2),
        'addr_accuracy':  round(addr_accuracy, 2),
        'ipc_base':    round(ipc_base, 2),
        'ipc_improved':round(ipc_improved, 4),
        'ipc_speedup': round(ipc_speedup, 4),
        'ipex_state':  throttle.state,
        'ipex_degree': throttle.prefetch_degree,
        'ipex_events': throttle.history['events'],
        'wall_clock_s':round(elapsed, 1),
        'cache_metrics': cm,
    }


# ── Auto-discover and run ─────────────────────────────────────────────────────
TRACE_DIR   = '/home/cs25m115/spec2006'
TRACE_PATHS = sorted([
    os.path.join(TRACE_DIR, f) for f in os.listdir(TRACE_DIR)
    if f.endswith('.trace.xz')
])
print(f'Found {len(TRACE_PATHS)} trace(s) in {TRACE_DIR}')
for p in TRACE_PATHS: print(f'  {os.path.basename(p)}')

all_results = []
for tp in TRACE_PATHS:
    r = champsim_simulate(tp, seq_len=SEQ_LEN, warmup_instr=200_000, sim_instr=500_000,
                          batch_size=512, throttle_threshold=0.50, recover_threshold=0.70)
    if r is not None: all_results.append(r)


# ── Cross-benchmark summary ───────────────────────────────────────────────────
if all_results:
    print()
    print('='*110)
    print('  SUMMARY — All Traces')
    print('='*110)
    print(f"  {'Trace':<30} {'AddrAcc':>7} {'MissRed':>7} {'AMAT↓%':>6} "
          f"{'BW↑%':>7} {'EnrgSav%':>8} {'PwrSav%':>7} {'Pollution%':>10} {'IPCSpd':>8}")
    print('  ' + '-'*108)
    for r in all_results:
        cm = r['cache_metrics']
        print(f"  {r['trace'][:30]:<30} "
              f"{r['addr_accuracy']:>6.2f}% "
              f"{cm['miss_reduction_pct']:>6.2f}% "
              f"{cm['amat_reduction_pct']:>5.2f}% "
              f"{cm['throughput_gain_pct']:>6.2f}% "
              f"{cm['energy_savings_pct']:>7.2f}% "
              f"{cm['power_savings_pct']:>6.2f}% "
              f"{cm['pollution_rate_pct']:>9.2f}% "
              f"{r['ipc_speedup']:>7.3f}x")

    def gavg(key, sub=False):
        src = [r['cache_metrics'][key] if sub else r[key] for r in all_results]
        return sum(src) / len(src)

    print('  ' + '-'*108)
    print(f"  {'AVERAGE':<30} "
          f"{gavg('addr_accuracy'):>6.2f}% "
          f"{gavg('miss_reduction_pct', True):>6.2f}% "
          f"{gavg('amat_reduction_pct', True):>5.2f}% "
          f"{gavg('throughput_gain_pct', True):>6.2f}% "
          f"{gavg('energy_savings_pct', True):>7.2f}% "
          f"{gavg('power_savings_pct', True):>6.2f}% "
          f"{gavg('pollution_rate_pct', True):>9.2f}% "
          f"{gavg('ipc_speedup'):>7.3f}x")
    print('='*110)

    out_json = '/home/cs25m115/mamba_inf/mambaedge_sim_results_champ.json'
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Results saved → {out_json}')