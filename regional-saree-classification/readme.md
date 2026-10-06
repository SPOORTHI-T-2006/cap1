# Regional Saree Classification

Deep-learning image classifier that recognises the **regional draping / traditional style** of a saree
from a photo. Four classes: **Assamese, Coorg (Kodagu), Gujarati, Maharashtrian**.

Model: ResNet18 pretrained on ImageNet, fine-tuned in two stages (PyTorch).

## Results (held-out test set, 116 images, 29 per class)

| Metric | Value |
|---|---|
| Accuracy | 97.41 % |
| Precision / Recall / F1 (macro) | 0.974 / 0.974 / 0.974 |
| ROC AUC (one-vs-rest, macro) | 0.9997 |

Plots and metrics are in [`results/`](results/): confusion matrix, ROC curves, training curves, `metrics.json`.

## Project structure

```
saree_project/
├── train.py            # data cleaning, split, training, evaluation
├── predict.py          # show predictions on new images
├── requirements.txt
├── results/            # best_model.pth, plots, metrics, split lists
├── data/               # (not in repo) one sub-folder per class
└── test_images/        # (not in repo) images you want to test
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

Put images in `data/<class_name>/` (one folder per class), then:

```bash
python train.py --data_dir data
```

The script skips corrupt files and exact duplicates, uses at most 200 good images per class
(`--per_class`), balances the classes, makes a stratified 70/15/15 train/val/test split,
trains, and evaluates once on the test set.

## Predict on new images

Put images in `test_images/` and run:

```bash
python predict.py              # one window per image (close it for the next)
python predict.py --mode grid  # all images in one window
```

## Notes

- Horizontal flip augmentation is off by default because mirroring changes the drape direction.
- The dataset is not included (web-collected images). Recreate the `data/` folder structure to retrain.
- The model only knows these 4 classes; any other image will still be assigned to one of them.
