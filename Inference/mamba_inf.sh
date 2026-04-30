#!/bin/bash
#SBATCH --job-name=mambaedge_infer
#SBATCH --partition=q3d                     # ← same partition as training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # inference = single process
#SBATCH --gres=gpu:1                        # 1 GPU is enough for inference
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00                      # 4 hours (all traces, cached = fast)
#SBATCH --output=logs/mambaedge_infer_%j.out
#SBATCH --error=logs/mambaedge_infer_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cs25m115@iittp.ac.in

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR="/home/cs25m115/mamba_inf"
TRACE_DIR="/home/cs25m115/spec2006"
CKPT="/home/cs25m115/mamba/checkpoints/mambaedge_best.pth"
INFER_SCRIPT="mambaedge_inference.py"
SIM_SCRIPT="champsim_simulate.py"
RESULTS_DIR="/home/cs25m115/mamba_inf/results"

# ── Conda env (same as training) ─────────────────────────────────────────────
CONDA_ENV="mamba_env"

echo "============================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURMD_NODENAME"
echo "  Start      : $(date)"
echo "  Checkpoint : $CKPT"
echo "  Trace dir  : $TRACE_DIR"
echo "============================================================"

# ── Activate conda ────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# ── Setup ─────────────────────────────────────────────────────────────────────
cd $WORK_DIR
mkdir -p logs results

# ── Verify checkpoint exists ──────────────────────────────────────────────────
if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    echo "  Make sure training completed and mambaedge_best.pth exists."
    exit 1
fi

# ── GPU info ──────────────────────────────────────────────────────────────────
echo ""
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
echo ""

# ── Environment variables ─────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# ── Count available traces ────────────────────────────────────────────────────
N_TRACES=$(ls $TRACE_DIR/*.trace.xz 2>/dev/null | wc -l)
N_CACHED=$(ls $TRACE_DIR/*.trace.xz.pages.npy 2>/dev/null | wc -l)
echo "Traces found : $N_TRACES"
echo "Cached       : $N_CACHED  (.pages.npy files)"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# MODE 1 — Quick sanity check (demo on 3 traces, no SLURM overhead)
# ════════════════════════════════════════════════════════════════════════════
echo "------------------------------------------------------------"
echo " STEP 1: Sanity check — run inference demo on 3 traces"
echo "------------------------------------------------------------"

# # Pick 3 traces (first, middle, last alphabetically)
# TRACES=( $(ls $TRACE_DIR/*.champsimtrace.xz | head -1)
#          $(ls $TRACE_DIR/*.champsimtrace.xz | awk 'NR==int(NR/2)')
#          $(ls $TRACE_DIR/*.champsimtrace.xz | tail -1) )
mapfile -t TRACES < <(printf '%s\n' "$TRACE_DIR"/*.trace.xz)


for TRACE in "${TRACES[@]}"; do
    echo ""
    echo "  >>> $(basename $TRACE)"
    python $WORK_DIR/$INFER_SCRIPT "$TRACE"
done

# ════════════════════════════════════════════════════════════════════════════
# MODE 2 — Full simulation across ALL traces (uses champsim_simulate.py)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "------------------------------------------------------------"
echo " STEP 2: Full ChampSim simulation across ALL $N_TRACES traces"
echo "------------------------------------------------------------"
echo ""

python $WORK_DIR/$SIM_SCRIPT

EXIT_CODE=$?

# ── Copy results to results dir ───────────────────────────────────────────────
if [ -f "$WORK_DIR/checkpoints/mambaedge_sim_results.json" ]; then
    cp "$WORK_DIR/checkpoints/mambaedge_sim_results.json" \
       "$RESULTS_DIR/mambaedge_sim_results_${SLURM_JOB_ID}.json"
    echo ""
    echo "Results copied → $RESULTS_DIR/mambaedge_sim_results_${SLURM_JOB_ID}.json"
fi

echo ""
echo "============================================================"
echo "  Finished : $(date)"
echo "  Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
