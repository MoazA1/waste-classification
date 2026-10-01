# Garbage Classification

A deep-learning project that classifies a single household-waste item into one of 12 categories.

| Phase | Notebook | Content |
|---|---|---|
| 1 — Proposal | `01_dataset_inspection.ipynb` (clean) / `01_dataset_inspection_Phase1_Results.ipynb` (executed) | Dataset inspection, class balance, planned split, metric. No model training. |
| 2 — Data pipeline + MLPs | `02_mlp_baselines.ipynb` | PyTorch data pipeline, shallow (2 hidden layers) and deep (5 hidden layers) MLPs, loss × optimizer × learning-rate experiments. |

## Problem definition

Incorrectly sorted waste contaminates recycling streams and increases the amount of manual inspection required. The proposed system represents a controlled sorting point in which one dominant waste item is photographed at a time.

- **Input:** one RGB image containing a dominant household-waste item.
- **Output:** one of 12 waste categories.
- **Task:** multiclass image classification.
- **Intended use:** assist a smart bin, conveyor-routing mechanism, or human sorting operator.
- **Operational decision:** select the appropriate waste-processing stream or request manual review.


## Dataset

- **Kaggle dataset:** [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- **Kaggle handle:** `mostafaabla/garbage-classification`
- **Expected labels:** battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, and white-glass.
- **Access:** downloaded from Kaggle by `kagglehub`; a Kaggle account or API authentication may be required depending on the runtime.
- **License:** Open Data Commons Open Database License (ODbL) 1.0, as listed for the Kaggle dataset. Users should also follow Kaggle's terms of use.

The notebook calculates the actual image count, class frequencies, percentages, image dimensions, and unreadable-file count from the downloaded dataset.

## Predeclared experimental plan

- **Split:** stratified 70% training, 15% validation, and 15% testing.
- **Random seed:** 42.
- **Single validation metric:** macro-averaged F1 score.

Macro F1 gives every class equal importance, which is appropriate because the dataset is imbalanced. The test split must remain untouched until final model evaluation in a later phase.

## Repository contents

```text
.
├── 01_dataset_inspection.ipynb                 # Phase 1 notebook (no outputs)
├── 01_dataset_inspection_Phase1_Results.ipynb  # Phase 1 notebook, executed in Colab
├── 02_mlp_baselines.ipynb                      # Phase 2 notebook
├── .gitignore
└── README.md
```

The Phase 1 notebook produces:

```text
artifacts/
├── class_distribution.csv
├── class_distribution.png
├── dataset_inventory.csv
├── dataset_summary.json
├── image_dimension_summary.csv
├── representative_samples.png
└── split_manifest.csv
```

## Running Phase 1 in Google Colab

1. Upload or open `01_dataset_inspection.ipynb` in Colab.
2. Select **Runtime → Run all**. A GPU is not required for this phase.
3. If Kaggle requests authentication, upload your Kaggle API token when prompted and rerun the download cell.
4. Download the generated `artifacts` directory or commit the generated files to the repository.

The notebook is rerunnable: it uses a fixed random seed and does not modify the downloaded dataset.

## Scope of Phase 1

Included: data download, file validation, actual sample inspection, class-balance table, representative sample grid, distribution plot, image-dimension summary, and planned split manifest.

Not included: neural-network selection, training, hyperparameter tuning, or test-set evaluation.


## Phase 2 — Data pipeline and MLP baselines

`02_mlp_baselines.ipynb` is self-contained. Every step is a reusable function or class, and each section header names the requirement it covers. The committed notebook is the executed version, with all outputs and a discussion of the results.

- **Pipeline:** Kaggle download → one sample per class → stratified 80/20 train/test split, then 80/20 train/validation (64/16/20 overall, seed 42) → resize to 64×64 → per-channel normalisation fitted on the training split only → flatten to 12,288 values → `LabelEncoder` fitted on training labels → tensors → `DataLoader`s.
- **Models:** one `MLP` class with configurable hidden layers. Shallow = `[512, 256]`, deep = `[512, 256, 128, 64, 32]`; each hidden layer is Linear → BatchNorm → ReLU → Dropout(0.3); 12 output logits.
- **Experiments:** 3 losses (cross-entropy, label smoothing 0.1, focal γ=2) × 3 optimizers (SGD+momentum, RMSprop, Adam) on the shallow MLP, then 4 learning rates for the best pair, then the deep MLP with the optimal settings, then a test-set comparison. L2 weight decay λ = 1e-4 throughout.
- **Selection metric:** validation macro-F1 (from the Phase 1 proposal). The test set is used only once, at the end.

### Running Phase 2 in Google Colab

1. Open `02_mlp_baselines.ipynb` in Colab and choose **Runtime → Change runtime type → GPU**.
2. Select **Runtime → Run all**. The first cell installs `kagglehub`; everything else is preinstalled in Colab.
3. Training runs 14 models of 40 epochs each; on a GPU this takes roughly 10–20 minutes.
4. Save the executed notebook (**File → Download → .ipynb**) so all outputs are visible.
