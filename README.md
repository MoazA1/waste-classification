# Garbage Classification — Proposal Support

This repository contains the dataset-inspection code for a proposed deep-learning system that classifies a single household-waste item into one of 12 categories. This phase intentionally contains **no model training**.

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
├── notebooks/
│   └── 01_dataset_inspection.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

The notebook produces:

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

## Running in Google Colab

1. Upload or open `notebooks/01_dataset_inspection.ipynb` in Colab.
2. Select **Runtime → Run all**. A GPU is not required for this phase.
3. If Kaggle requests authentication, upload your Kaggle API token when prompted and rerun the download cell.
4. Download the generated `artifacts` directory or commit the generated files to the repository.

The notebook is rerunnable: it uses a fixed random seed and does not modify the downloaded dataset.

## Scope of this phase

Included: data download, file validation, actual sample inspection, class-balance table, representative sample grid, distribution plot, image-dimension summary, and planned split manifest.

Not included: neural-network selection, training, hyperparameter tuning, or test-set evaluation.
