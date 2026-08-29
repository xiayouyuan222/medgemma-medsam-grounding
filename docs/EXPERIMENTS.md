# Experiment map

## Principal comparisons

| Experiment | Initialisation | Training data | Main option |
|---|---|---|---|
| TotalSeg-14 projector | MedGemma + MedSAM | TotalSeg-14 | `MODEL_VARIANT=projector` |
| TotalSeg-14 aligner | Projector checkpoint | TotalSeg-14 | `MODEL_VARIANT=aligner` |
| SLAKE-only aligner | MedGemma + MedSAM | SLAKE | `SOURCE=slake-only` |
| Transfer aligner | TotalSeg-14 aligner | SLAKE | `SOURCE=totalseg-transfer` |

The TotalSeg-14 and SLAKE evaluations use the same metric implementation in
`evaluate_medgemma_gold_phrase_seg.py`.

## Auxiliary-loss ablation

Use the projector checkpoint as `INIT_ADAPTER` and keep all other arguments
fixed.

```bash
# Continued-training control
sbatch --export=ALL,MODEL_VARIANT=projector,TAG=control,\
INIT_ADAPTER=$PROJECT_ROOT/outputs/PROJECTOR_CHECKPOINT,\
EPOCHS=10,LR=5e-6,EDGE_WEIGHT=0.0,SWITCH_WEIGHT=0.0 \
slurm/train_totalseg14.sbatch

# Boundary loss
sbatch --export=ALL,MODEL_VARIANT=projector,TAG=edge,\
INIT_ADAPTER=$PROJECT_ROOT/outputs/PROJECTOR_CHECKPOINT,\
EPOCHS=10,LR=5e-6,EDGE_WEIGHT=0.2,SWITCH_WEIGHT=0.0 \
slurm/train_totalseg14.sbatch

# Same-image distractor margin loss
sbatch --export=ALL,MODEL_VARIANT=projector,TAG=distractor,\
INIT_ADAPTER=$PROJECT_ROOT/outputs/PROJECTOR_CHECKPOINT,\
EPOCHS=10,LR=5e-6,EDGE_WEIGHT=0.0,SWITCH_WEIGHT=0.2 \
slurm/train_totalseg14.sbatch

# Combined losses
sbatch --export=ALL,MODEL_VARIANT=projector,TAG=combined,\
INIT_ADAPTER=$PROJECT_ROOT/outputs/PROJECTOR_CHECKPOINT,\
EPOCHS=10,LR=5e-6,EDGE_WEIGHT=0.2,SWITCH_WEIGHT=0.2 \
slurm/train_totalseg14.sbatch
```

The option named `switch-loss` in the implementation compares a prediction
with another region from the same image. It does not perform a second forward
pass with a swapped phrase, so the dissertation describes it as a same-image
distractor margin loss.

## Visual-path controls

Retraining without a real MedGemma image uses:

```bash
sbatch --export=ALL,MODEL_VARIANT=projector,TAG=single-vision,\
DISABLE_VLM_IMAGE=1 slurm/train_totalseg14.sbatch
```

The inference-time image interventions are implemented separately in
`test_dual_vision_necessity.py`.

## Historical files

`slurm/legacy/` contains the original cluster scripts, including smoke tests,
the two-GPU aligner run and early predicted-bounding-box experiments. They are
retained for traceability but are not the recommended public interface.
