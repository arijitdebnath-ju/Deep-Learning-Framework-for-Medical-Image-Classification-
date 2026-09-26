# MedMNIST Multi-Architecture Benchmark

Benchmarking four CNN/Transformer architectures — **ResNet-18**, **EfficientNet-B0**, **DenseNet-121**, and **MobileViT-S** — on four [MedMNIST v2](https://medmnist.com/) medical imaging datasets: **BreastMNIST**, **DermaMNIST**, **RetinaMNIST**, and **BloodMNIST**.

Each notebook runs the exact same training/evaluation pipeline; only the target dataset pair differs.

| Notebook | Datasets covered |
|---|---|
| `breastdermaversion.ipynb` | BreastMNIST, DermaMNIST |
| `bloodretinaversion.ipynb` | RetinaMNIST, BloodMNIST |

## What the pipeline does

For each dataset, the notebook:

1. Downloads the dataset via the `medmnist` package and splits the official training set into **85% train / 15% validation** (the official test split is kept untouched).
2. Converts grayscale images to RGB, resizes to `224x224`, applies data augmentation (random horizontal flip, ±15° rotation) for training, and normalizes with ImageNet statistics.
3. Trains all four architectures from ImageNet-pretrained weights, one after another:
   - `resnet18`
   - `efficientnet_b0`
   - `densenet121`
   - `mobilevit_s` (via [`timm`](https://github.com/huggingface/pytorch-image-models))
4. Optimizes with `AdamW` + cosine annealing LR schedule, tracks the best validation-accuracy checkpoint per model, and restores those weights before final evaluation.
5. Evaluates every model on the train, validation, and test splits, reporting **accuracy, precision, recall, and F1 (weighted)**.
6. Plots training/validation loss and accuracy curves per architecture, plus a grouped bar chart comparing test-set metrics across architectures.

## Setup

```bash
pip install medmnist torch torchvision scikit-learn tqdm matplotlib numpy timm
```

A CUDA-capable GPU is strongly recommended — each notebook trains 4 architectures x 2 datasets x 50 epochs.

## Configuration

Key hyperparameters (set at the top of the pipeline cell):

```python
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
```

A fixed seed (`42`) is set across `random`, `numpy`, and `torch` for reproducibility.

## Results

### BreastMNIST / DermaMNIST (`breastdermaversion.ipynb`)

**BreastMNIST — Test set**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| ResNet-18 | 0.8462 | 0.8413 | 0.8462 | 0.8422 |
| EfficientNet-B0 | 0.8782 | 0.8760 | 0.8782 | 0.8732 |
| DenseNet-121 | 0.8718 | 0.8685 | 0.8718 | 0.8685 |
| MobileViT-S | 0.8718 | 0.8686 | 0.8718 | 0.8672 |

**DermaMNIST — Test set**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| ResNet-18 | 0.7671 | 0.7526 | 0.7671 | 0.7561 |
| EfficientNet-B0 | 0.7970 | 0.7856 | 0.7970 | 0.7889 |
| DenseNet-121 | 0.7796 | 0.7628 | 0.7796 | 0.7676 |
| MobileViT-S | 0.7880 | 0.7790 | 0.7880 | 0.7811 |

### RetinaMNIST / BloodMNIST (`bloodretinaversion.ipynb`)

**RetinaMNIST — Test set**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| ResNet-18 | 0.4850 | 0.4530 | 0.4850 | 0.4645 |
| EfficientNet-B0 | 0.5050 | 0.4933 | 0.5050 | 0.4982 |
| DenseNet-121 | 0.4625 | 0.4612 | 0.4625 | 0.4604 |
| MobileViT-S | 0.4975 | 0.4775 | 0.4975 | 0.4851 |

**BloodMNIST — Test set**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| ResNet-18 | 0.9734 | 0.9736 | 0.9734 | 0.9734 |
| EfficientNet-B0 | 0.9790 | 0.9792 | 0.9790 | 0.9790 |
| DenseNet-121 | 0.9784 | 0.9785 | 0.9784 | 0.9784 |
| MobileViT-S | 0.9825 | 0.9826 | 0.9825 | 0.9825 |

Full training/validation curves per model, and per-split metric tables (train/val/test), are printed and plotted inline in each notebook.

### Observations

- All four models reach ~99–100% **training** accuracy, indicating the models fit the training data very well (and, on the smaller datasets, likely overfit).
- **BloodMNIST** (a relatively large, well-balanced 8-class dataset) achieves the strongest and most consistent generalization across all architectures (~97–98% test accuracy).
- **RetinaMNIST** (a small, ordinal 5-class dataset) is the hardest task, with test accuracy around 46–51% for every architecture — far below its training accuracy, reflecting the small dataset size and class imbalance.
- **EfficientNet-B0** and **MobileViT-S** tend to edge out ResNet-18 and DenseNet-121 on test-set metrics across most datasets, though the gap is small.

## Repository structure

```
.
├── breastdermaversion.ipynb   # BreastMNIST + DermaMNIST pipeline
├── bloodretinaversion.ipynb   # RetinaMNIST + BloodMNIST pipeline
└── README.md
```

## Notes & limitations

- Models are trained independently per dataset/architecture — there is no cross-dataset transfer.
- The train/validation split uses `torch.utils.data.random_split` with a fixed seed; the official MedMNIST test split is used only for final evaluation, never for model selection.
- No class-imbalance handling (e.g. weighted loss, oversampling) is applied, which likely affects results on imbalanced datasets like DermaMNIST and RetinaMNIST.
- Fixed hyperparameters are shared across all architectures/datasets — no per-model tuning was performed.

## Acknowledgments

- [MedMNIST v2](https://medmnist.com/) — Yang et al., *"MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification"*.
- [`timm`](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models) for the MobileViT implementation.
- Torchvision pretrained ImageNet weights for ResNet, EfficientNet, and DenseNet.
