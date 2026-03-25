#!/bin/bash
#SBATCH --job-name=single_run
#SBATCH --output=results/single_run-%j.out
#SBATCH --error=results/single_run-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=long

# SLURM_SUBMIT_DIR is set by SLURM to the directory where sbatch was invoked.
# Submit from the project root:  sbatch scripts/run_single_notebook.sh
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

# =============================================================================
# Experiment configuration — single source of truth for all hyperparameters.
# Individual variables can be overridden *after* this block.
# =============================================================================

# ── Experiment identity ───────────────────────────────────────────────────────
EXPERIMENT_SETTING="1"   # 1 = Scenario A (color vs digit), 2 = Scenario B (digit vs watermark)
SEED="5"

# ── Network & optimisation ────────────────────────────────────────────────────
NETWORK_NAME="MLP"       # MLP | CNN
N_OUTPUTS="256"
OPTIMIZER_NAME="adamw"   # sgd | adam | adamw
WEIGHT_DECAY="0.0001"
BATCH_SIZE="64"

# ── Dataset – spurious correlation ────────────────────────────────────────────
SPUR_PROB="0.1"          # Fraction of samples in environment 1 (0.1 → imbalanced)
FLIP_PROB="0.09"         # Label-noise probability
ENV_NOISINESS="0.05"     # Watermark corruption rate (Setting 2)
ATTR_PROB="0.5"          # Attribute balance (keep at 0.5)
CMNIST_DIGITS_PER_CLASS="5"  # Number of distinct digit shapes per class (1–5)
UNINFORMATIVE_MAJORITY="False"  # Setting 1: randomise colour in the majority env

# ── Dataset – watermark (Setting 2 only) ──────────────────────────────────────
WATERMARK_BANK_SIZE="2"  # Distinct watermark patterns per environment (controls complexity K)

# ── Misc ──────────────────────────────────────────────────────────────────────
DEBUG_MODE="False"
LOAD_PRETRAINED="False"

# ── Per-run overrides (uncomment and edit as needed) ─────────────────────────
# EXPERIMENT_SETTING="2"
# SPUR_PROB="0.5"
# WATERMARK_BANK_SIZE="50"
# SEED="42"
# -----------------------------------------------------------------------------

KERNEL_NAME="invariant-bench"

# Timestamp-based result folder
RESULT_FOLDER=$(date +%Y%m%d-%H%M%S)
mkdir -p "${PROJECT_ROOT}/results/${RESULT_FOLDER}"

# Activate environment
module load cuda/11.8
source "${PROJECT_ROOT}/.venv/bin/activate"

# Export all config variables so the notebook can read them via os.environ
export EXPERIMENT_SETTING SEED
export NETWORK_NAME N_OUTPUTS OPTIMIZER_NAME WEIGHT_DECAY BATCH_SIZE
export SPUR_PROB FLIP_PROB ENV_NOISINESS ATTR_PROB CMNIST_DIGITS_PER_CLASS UNINFORMATIVE_MAJORITY
export WATERMARK_BANK_SIZE
export DEBUG_MODE LOAD_PRETRAINED
export RESULT_FOLDER

echo "Project root       : ${PROJECT_ROOT}"
echo "Experiment setting : ${EXPERIMENT_SETTING}"
echo "Seed               : ${SEED}"
echo "Network            : ${NETWORK_NAME} (n_outputs=${N_OUTPUTS})"
echo "Optimizer          : ${OPTIMIZER_NAME} (wd=${WEIGHT_DECAY})"
echo "spur_prob          : ${SPUR_PROB}  flip_prob=${FLIP_PROB}  env_noise=${ENV_NOISINESS}"
echo "Watermark bank size: ${WATERMARK_BANK_SIZE}"
echo "Results folder     : ${RESULT_FOLDER}"

cd "${PROJECT_ROOT}"
"${PROJECT_ROOT}/.venv/bin/jupyter" nbconvert \
    --to notebook --execute main.ipynb \
    --ExecutePreprocessor.kernel_name="${KERNEL_NAME}" \
    --ExecutePreprocessor.timeout=0 \
    --output-dir "${PROJECT_ROOT}/results/${RESULT_FOLDER}" \
    --output executed_main_notebook.ipynb
