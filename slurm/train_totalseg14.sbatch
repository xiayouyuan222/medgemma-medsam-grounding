#!/bin/bash
#SBATCH --job-name=ts14-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/ts14-train-%j.out
#SBATCH --error=logs/ts14-train-%j.err

set -euo pipefail

: "${MODEL_VARIANT:?Set MODEL_VARIANT to projector, aligner, or direct-mask}"
: "${TAG:?Set TAG to a unique experiment name}"

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/conda/medgemma/bin/python}
INIT_ADAPTER=${INIT_ADAPTER:-}
EPOCHS=${EPOCHS:-3}
LR=${LR:-2e-5}
EDGE_WEIGHT=${EDGE_WEIGHT:-0.0}
SWITCH_WEIGHT=${SWITCH_WEIGHT:-0.0}
DISABLE_VLM_IMAGE=${DISABLE_VLM_IMAGE:-0}
EVAL_STRATEGY=${EVAL_STRATEGY:-epoch}
EVAL_STEPS=${EVAL_STEPS:-500}
SAVE_STEPS=${SAVE_STEPS:-500}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}
EARLY_STOPPING_THRESHOLD=${EARLY_STOPPING_THRESHOLD:-0.002}

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"

export HF_HOME=${HF_HOME:-$PROJECT_ROOT/hf_cache}
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA_ARGS=()
case "$MODEL_VARIANT" in
  projector)
    EXTRA_ARGS+=(--grounding-decoder-mode medsam --num-prompt-tokens 1)
    ;;
  aligner)
    EXTRA_ARGS+=(
      --grounding-decoder-mode aligner-medsam
      --aligner-heads 8
      --spatial-merge-factor 2
      --spatial-topk 64
      --swiglu-expansion 4
      --aligner-dropout 0.0
    )
    ;;
  direct-mask)
    EXTRA_ARGS+=(
      --grounding-decoder-mode direct-mask
      --aligner-heads 8
      --spatial-merge-factor 2
      --spatial-topk 64
      --swiglu-expansion 4
    )
    ;;
  *)
    echo "Unknown MODEL_VARIANT=$MODEL_VARIANT" >&2
    exit 2
    ;;
esac

if [[ -n "$INIT_ADAPTER" ]]; then
  EXTRA_ARGS+=(--init-adapter "$INIT_ADAPTER")
fi
if [[ "$DISABLE_VLM_IMAGE" == "1" ]]; then
  EXTRA_ARGS+=(--disable-vlm-image)
fi
if (( EARLY_STOPPING_PATIENCE > 0 )); then
  EXTRA_ARGS+=(
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --early-stopping-threshold "$EARLY_STOPPING_THRESHOLD"
  )
fi

"$PYTHON_BIN" -u "$PROJECT_ROOT/code/train_medgemma_vividmed_lite_seg.py" \
  --model "$PROJECT_ROOT/models/medgemma-4b-it" \
  --sam-checkpoint "$PROJECT_ROOT/checkpoints/medsam_vit_b.pth" \
  --data-root "$PROJECT_ROOT/data/totalsegmentator_2d_100" \
  --train-file train.jsonl \
  --val-file validation.jsonl \
  --output-dir "$PROJECT_ROOT/outputs/${TAG}-${SLURM_JOB_ID}" \
  --training-mode phrase-only \
  --text-loss-weight 0.0 \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --dice-loss-weight 1.0 \
  --bce-loss-weight 1.0 \
  --focal-loss-weight 0.0 \
  --edge-loss-weight "$EDGE_WEIGHT" \
  --edge-kernel-size 3 \
  --switch-loss-weight "$SWITCH_WEIGHT" \
  --switch-margin 0.2 \
  --gradient-accumulation-steps 8 \
  --eval-strategy "$EVAL_STRATEGY" \
  --eval-steps "$EVAL_STEPS" \
  --save-steps "$SAVE_STEPS" \
  "${EXTRA_ARGS[@]}"
