#!/bin/bash
#SBATCH --job-name=mambaedge_sim
#SBATCH --partition=q3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/mambaedge_sim_no_ipex_%j.out
#SBATCH --error=logs/mambaedge_sim_n0_ipex_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cs25m115@iittp.ac.in

# ═══════════════════════════════════════════════════════════════
# PATHS — edit only if you move files
# ═══════════════════════════════════════════════════════════════
CONDA_ENV="mamba_env"
WORK_DIR="/home/cs25m115/mamba_inf"
TRACE_DIR="/home/cs25m115/spec2006"
CKPT="/home/cs25m115/mamba/checkpoints/mambaedge_best.pth"
SCRIPT="${WORK_DIR}/champsin_sim_no_ipex.py"
OUT_JSON="${WORK_DIR}/mambaedge_sim_no_ipex_results_champ.json"
# ═══════════════════════════════════════════════════════════════

echo "============================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURMD_NODENAME"
echo "  Start      : $(date)"
echo "  Script     : $SCRIPT"
echo "  Checkpoint : $CKPT"
echo "  Trace dir  : $TRACE_DIR"
echo "============================================================"

# ── Activate conda ────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# ── Setup ─────────────────────────────────────────────────────
cd $WORK_DIR
mkdir -p logs

# ── Sanity checks ─────────────────────────────────────────────
if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    exit 1
fi

if [ ! -d "$TRACE_DIR" ]; then
    echo "[ERROR] Trace directory not found: $TRACE_DIR"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "[ERROR] Script not found: $SCRIPT"
    exit 1
fi

# ── GPU info ──────────────────────────────────────────────────
echo ""
nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
           --format=csv,noheader
echo ""

# ── Count traces ──────────────────────────────────────────────
N_TRACES=$(ls $TRACE_DIR/*.trace.xz 2>/dev/null | wc -l)
echo "Traces found (.trace.xz) : $N_TRACES"
if [ "$N_TRACES" -eq 0 ]; then
    echo "[ERROR] No .trace.xz files found in $TRACE_DIR"
    exit 1
fi
echo ""

# ── Environment ───────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# ── Run simulation ────────────────────────────────────────────
echo "------------------------------------------------------------"
echo " Running ChampSim simulation on $N_TRACES trace(s) ..."
echo "------------------------------------------------------------"
echo ""

python $SCRIPT

EXIT_CODE=$?

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Finished  : $(date)"
echo "  Exit code : $EXIT_CODE"
if [ -f "$OUT_JSON" ]; then
    echo "  Results   : $OUT_JSON"
    echo ""
    echo "  Quick summary (addr_accuracy | ipc_speedup per trace):"
    python3 -c "
import json
with open('$OUT_JSON') as f:
    results = json.load(f)
print(f\"  {'Trace':<40} {'Addr Acc':>8} {'IPC Spdup':>10}\")
print('  ' + '-'*60)
for r in results:
    print(f\"  {r['trace'][:40]:<40} {r['addr_accuracy']:>7.2f}%  {r['ipc_speedup']:>9.3f}x\")
avg_acc = sum(r['addr_accuracy'] for r in results)/len(results)
avg_spd = sum(r['ipc_speedup']   for r in results)/len(results)
print('  ' + '-'*60)
print(f\"  {'AVERAGE':<40} {avg_acc:>7.2f}%  {avg_spd:>9.3f}x\")
" 2>/dev/null || echo "  (install python3 json to see summary)"
fi
echo "============================================================"

exit $EXIT_CODE
