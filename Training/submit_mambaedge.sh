#!/bin/bash
#SBATCH --job-name=mambaedge_train
#SBATCH --partition=q5h                # ← EDIT: your partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3             # ← EDIT: number of GPUs (must match --gres)
#SBATCH --gres=gpu:3                    # ← EDIT: must match ntasks-per-node
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=logs/mambaedge_%j.out
#SBATCH --error=logs/mambaedge_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cs25m115@iittp.ac.in    # ← EDIT: your email

# ── How many GPUs — hardcoded, NEVER use $SLURM_NTASKS_PER_NODE (can be empty)
NUM_GPUS=3                              # ← EDIT: same as --gres=gpu:N above

# ── Conda environment ─────────────────────────────────────────────────────────
CONDA_ENV="mamba_env"                    # your env (cs25m115 uses detr_env)

# ── Project directory ─────────────────────────────────────────────────────────
WORK_DIR="/home/cs25m115/mamba"              # ← EDIT: where mambaedge_dgx_train.py lives
SCRIPT="mambaedge_dgx_train.py"

echo "============================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURMD_NODENAME"
echo "  Num GPUs   : $NUM_GPUS"
echo "  Start      : $(date)"
echo "============================================================"

# ── Activate conda ────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_env

# ── Move to working dir and create output folders ─────────────────────────────
cd $WORK_DIR
mkdir -p logs checkpoints

# ── Print GPU info ────────────────────────────────────────────────────────────
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ── Distributed environment variables ────────────────────────────────────────
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0  # don't let watchdog SIGABRT on slow barriers
export TORCH_NCCL_BLOCKING_WAIT=0          # non-blocking wait
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1               # disable InfiniBand (not present on most DGX nodes)
export NCCL_P2P_DISABLE=0             # keep NVLink P2P on
export NCCL_SOCKET_IFNAME=lo          # use loopback — avoids interface detection hangs
export CUDA_VISIBLE_DEVICES=0,1,2     # explicitly expose all 3 GPUs


# ── Launch with torchrun (--standalone avoids needing rdzv server) ────────────
echo "Launching torchrun with $NUM_GPUS GPU(s) ..."
echo ""

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $WORK_DIR/$SCRIPT

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Finished : $(date)"
echo "  Exit code: $EXIT_CODE"
echo "============================================================"
echo ""
echo "Checkpoint files:"
ls -lh $WORK_DIR/checkpoints/ 2>/dev/null || echo "  (none found)"

exit $EXIT_CODE