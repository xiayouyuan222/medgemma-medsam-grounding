# Code inventory

## Included in the dissertation package

The `code/` directory contains the final data, model, training and evaluation
path used for the reported phrase-grounding experiments. The `analysis/`
directory contains scripts used to aggregate metrics and prepare dissertation
figures. Original Slurm files are retained under `slurm/legacy/` so that exact
job settings remain auditable.

## Kept outside the public package

The following workspace groups are exploratory or document-production tools and
are not required to reproduce the final experiments:

- early region-classification, VQA and bounding-box LoRA prototypes;
- Qwen2-VL and translation tests;
- DOCX construction, translation and formatting scripts;
- temporary smoke-test utilities and contact-sheet generators;
- downloaded models, raw datasets, checkpoints, predictions and logs;
- duplicated `exports/`, `server/` and report snapshots.

Excluding these files avoids presenting abandoned experiments as part of the
final method. They remain in the original research workspace and have not been
deleted.

