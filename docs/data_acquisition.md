# Getting the data

The data is not in this repository. PAN data is for research use only and must not be redistributed, so `data/` is in `.gitignore`.

## PAN 2025 style change detection (main data)

This is the only data DeSAF is trained on.

1. Register at [tira.io](https://www.tira.io).
2. Request access on Zenodo with the same email: [zenodo.org/records/14891299](https://zenodo.org/records/14891299).
3. Unpack the train, validation and test splits (the test labels were released after the competition) into:

```
data/raw/pan2025_full/
  easy/
    train/        problem-1.txt, truth-problem-1.json, ...
    validation/
    test/
  medium/
    train/
    validation/
    test/
  hard/
    train/
    validation/
    test/
```

Each `problem-X.txt` has one sentence per line. Each `truth-problem-X.json` has a `changes` list with one 0 or 1 per pair of neighbouring sentences.

`src/utils/data_loader.py` reads this folder with `load_pan2025_full("data/raw", tier, split)`. Most scripts take `--data-root data/raw`; a few (for example `construction_experiment.py`) take the `pan2025_full` folder itself, which their docstrings say.

Size: 4,200 training, 900 validation and 900 test documents per level.

## PAN 2026 (one comparison only)

Used only to compare the direction of the writing-style clues between the 2025 and 2026 Hard levels. Not used for training. Get it from the PAN 2026 multi-author writing style analysis page (Zenodo, access through TIRA) and pass its folder with `--data-root`, for example:

```
python scripts/pan_feature_direction.py --data-root "D:\mawsa26-pan-zenodo" --name pan2026
```

## Reuters C50 (cross-domain check)

News articles by 50 authors, used to build multi-author documents from naturally written text. Download "Reuter_50_50" from the UCI Machine Learning Repository and pass the folder that contains `C50train/` and `C50test/`:

```
python scripts/c50_multiauthor_test.py --c50-dir "C:\path\to\reuter+50+50"
```

## What is never committed

`data/`, `checkpoints/`, `results/`, model files (`*.pt`, `*.npz`) and the virtual environment. See `.gitignore`.
