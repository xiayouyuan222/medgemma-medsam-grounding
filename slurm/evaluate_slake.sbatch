#!/bin/bash
#SBATCH --job-name=eval-slake
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=logs/eval-slake-%j.out
#SBATCH --error=logs/eval-slake-%j.err

set -euo pipefail

: "${ADAPTER:?Set ADAPTER to a completed checkpoint directory}"
: "${TAG:?Set TAG to the experiment name}"

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/conda/medgemma/bin/python}
DATA_ROOT="$PROJECT_ROOT/data/vividmed_lite_seg_package/slake_segmentation"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/hf_cache}
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

for SPLIT in validation test; do
  "$PYTHON_BIN" -u "$PROJECT_ROOT/code/evaluate_medgemma_gold_phrase_seg.py" \
    --model "$PROJECT_ROOT/models/medgemma-4b-it" \
    --adapter "$ADAPTER" \
    --sam-checkpoint "$PROJECT_ROOT/checkpoints/medsam_vit_b.pth" \
    --data-root "$DATA_ROOT" \
    --test-file "${SPLIT}.jsonl" \
    --output "$PROJECT_ROOT/outputs/${TAG}-${SPLIT}.jsonl" \
    --threshold 0.6 \
    --prompt-mode phrase-only
done

"$PYTHON_BIN" -u "$PROJECT_ROOT/code/test_visual_grounding_causality.py" \
  --model "$PROJECT_ROOT/models/medgemma-4b-it" \
  --adapter "$ADAPTER" \
  --sam-checkpoint "$PROJECT_ROOT/checkpoints/medsam_vit_b.pth" \
  --data-root "$DATA_ROOT" \
  --test-file test.jsonl \
  --output-dir "$PROJECT_ROOT/outputs/grounding-${TAG}-test" \
  --threshold 0.6 \
  --prompt-mode phrase-only

