#!/bin/bash
#SBATCH --job-name=slake-aligner
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slake-aligner-%j.out
#SBATCH --error=logs/slake-aligner-%j.err

set -euo pipefail

: "${SOURCE:?Set SOURCE to slake-only or totalseg-transfer}"
: "${TAG:?Set TAG to a unique experiment name}"

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/conda/medgemma/bin/python}
INIT_ADAPTER=${INIT_ADAPTER:-}
LR=${LR:-5e-6}
EPOCHS=${EPOCHS:-10}

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"

export HF_HOME=${HF_HOME:-$PROJECT_ROOT/hf_cache}
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INIT_ARGS=()
case "$SOURCE" in
  slake-only)
    ;;
  totalseg-transfer)
    : "${INIT_ADAPTER:?Set INIT_ADAPTER for the transfer experiment}"
    test -f "$INIT_ADAPTER/semantic_spatial_aligner.pt" || {
      echo "INIT_ADAPTER is not a completed aligner checkpoint" >&2
      exit 2
    }
    INIT_ARGS+=(--init-adapter "$INIT_ADAPTER")
    ;;
  *)
    echo "Unknown SOURCE=$SOURCE" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" -u "$PROJECT_ROOT/code/train_medgemma_vividmed_lite_seg.py" \
  --model "$PROJECT_ROOT/models/medgemma-4b-it" \
  "${INIT_ARGS[@]}" \
  --sam-checkpoint "$PROJECT_ROOT/checkpoints/medsam_vit_b.pth" \
  --data-root "$PROJECT_ROOT/data/vividmed_lite_seg_package/slake_segmentation" \
  --train-file train.jsonl \
  --val-file validation.jsonl \
  --output-dir "$PROJECT_ROOT/outputs/${TAG}-${SLURM_JOB_ID}" \
  --training-mode phrase-only \
  --text-loss-weight 0.0 \
  --grounding-decoder-mode aligner-medsam \
  --aligner-heads 8 \
  --spatial-merge-factor 2 \
  --spatial-topk 64 \
  --swiglu-expansion 4 \
  --aligner-dropout 0.0 \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --dice-loss-weight 1.0 \
  --bce-loss-weight 1.0 \
  --focal-loss-weight 0.0 \
  --edge-loss-weight 0.0 \
  --switch-loss-weight 0.0 \
  --gradient-accumulation-steps 8 \
  --eval-strategy epoch \
  --save-steps 25 \
  --early-stopping-patience 3 \
  --early-stopping-threshold 0.002

