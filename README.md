# Medical Phrase Grounding with MedGemma and MedSAM

This repository contains the code used for the dissertation experiments on
teacher-forced medical phrase grounding. The main pipeline connects contextual
phrase states from MedGemma-4B-IT to a MedSAM ViT-B mask decoder through either
a single-vector projector or a gated semantic-spatial aligner.

## Repository layout

```text
code/       Data preparation, model components, training, inference and evaluation
analysis/   Result aggregation and figure-generation utilities
slurm/      Portable Slurm entry points
slurm/legacy/
            Original job files retained as an experiment record
docs/       Experiment map and notes on excluded exploratory code
```

Model checkpoints, datasets, generated masks and logs are deliberately excluded.

## Main entry points

- `code/prepare_totalsegmentator_grounding.py`: construct patient-disjoint
  TotalSeg-14 JSONL splits from TotalSegmentator volumes.
- `code/prepare_slake_segmentation.py`: construct the English SLAKE phrase-mask
  subset used by the experiments.
- `code/train_medgemma_vividmed_lite_seg.py`: train the projector, aligner,
  direct-mask and auxiliary-loss variants.
- `code/evaluate_medgemma_gold_phrase_seg.py`: compute sample-mean IoU, Dice and
  IoU@0.5 with the teacher-forced phrase protocol.
- `code/test_visual_grounding_causality.py`: evaluate correct, swapped, zero and
  random phrase-state interventions.
- `code/test_dual_vision_necessity.py`: run image-path interventions without
  retraining.
- `code/evaluate_medsam_gt_box.py`: evaluate the oracle bounding-box reference.

## Expected project paths

The Slurm scripts use `PROJECT_ROOT` as the repository root and expect the
following external assets:

```text
$PROJECT_ROOT/models/medgemma-4b-it/
$PROJECT_ROOT/checkpoints/medsam_vit_b.pth
$PROJECT_ROOT/data/totalsegmentator_2d_100/
$PROJECT_ROOT/data/vividmed_lite_seg_package/slake_segmentation/
$PROJECT_ROOT/outputs/
$PROJECT_ROOT/logs/
```

Set the Python interpreter with `PYTHON_BIN`. If it is omitted, the scripts use
`$PROJECT_ROOT/conda/medgemma/bin/python`.

```bash
export PROJECT_ROOT=/mnt/scratch/USER/vlm
export PYTHON_BIN=$PROJECT_ROOT/conda/medgemma/bin/python
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"
```

## Reproduce the principal experiments

Train the TotalSeg-14 projector baseline from the pretrained model components:

```bash
sbatch --export=ALL,MODEL_VARIANT=projector,TAG=ts14-projector \
  slurm/train_totalseg14.sbatch
```

Train the semantic-spatial aligner. `INIT_ADAPTER` should identify the trained
projector checkpoint used for continued training in the reported experiment.

```bash
sbatch --export=ALL,MODEL_VARIANT=aligner,TAG=ts14-aligner,\
INIT_ADAPTER=$PROJECT_ROOT/outputs/PROJECTOR_CHECKPOINT,EPOCHS=10,LR=5e-6,\
EVAL_STRATEGY=steps,EVAL_STEPS=250,SAVE_STEPS=250,\
EARLY_STOPPING_PATIENCE=4 \
  slurm/train_totalseg14.sbatch
```

Evaluate TotalSeg-14 validation and test splits:

```bash
sbatch --export=ALL,ADAPTER=$PROJECT_ROOT/outputs/ALIGNER_CHECKPOINT,\
TAG=ts14-aligner slurm/evaluate_totalseg14.sbatch
```

Run the matched SLAKE-only experiment:

```bash
sbatch --export=ALL,SOURCE=slake-only,TAG=slake-only \
  slurm/train_slake.sbatch
```

Run TotalSeg-14 to SLAKE transfer with the same aligner architecture:

```bash
sbatch --export=ALL,SOURCE=totalseg-transfer,TAG=slake-transfer,\
INIT_ADAPTER=$PROJECT_ROOT/outputs/ALIGNER_CHECKPOINT \
  slurm/train_slake.sbatch
```

Evaluate either SLAKE checkpoint:

```bash
sbatch --export=ALL,ADAPTER=$PROJECT_ROOT/outputs/SLAKE_CHECKPOINT,\
TAG=slake-transfer slurm/evaluate_slake.sbatch
```

Commands for edge, same-image distractor and branch-retraining controls are in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Precision and task scope

MedGemma is loaded in bfloat16 and adapted with LoRA. The implementation does
not use 4-bit quantisation. The principal evaluator supplies the reference
phrase in a structured teacher-forced sequence; it does not evaluate free-form
phrase generation.

## Reproducibility notes

- TotalSegmentator splitting is deterministic and patient-disjoint.
- The source data and pretrained checkpoints are not redistributed.
- The reported threshold is `0.6` unless an experiment states otherwise.
- Legacy Slurm files retain the exact server paths and job settings used during
  development. The portable scripts are the recommended public entry points.
