# UBCOCEAN - Batch ML Pipelines for Ovarian Cancer WSI Image Processing

## Overview

This repo contains an exploratory ML workflow for the UBC-OCEAN dataset that focuses on batch-style, data-intensive image processing. The core challenge is working with very large pathology slide images by chunking them into smaller tiles, filtering out mostly-empty/background tiles, and running training and inference in a reproducible way.

The project is organized around:
- Dataset download utilities (Kaggle CLI)
- Tiling and pruning pipelines for large images
- Training and inference notebooks (including PyTorch Lightning-based workflows)
- Small supporting Python modules for models and tiling utilities

## What this demonstrates

- Batch-oriented preprocessing of multi-gigabyte image artifacts via tiling and staged execution
- Practical handling of memory and I/O constraints when working with large PNG slide images
- GPU-backed training and evaluation workflows using PyTorch / PyTorch Lightning
- Reproducible experimentation with clear separation between preprocessing, training, and inference

## Repository contents

### Notebooks

- `ubc-ocean-jpeg-dataset-pipeline.ipynb`
  - Dataset-oriented pipeline work and preprocessing exploration.

- `train.ipynb` / `inference.ipynb`
  - Baseline training and inference entrypoints.

- `cancer-subtype-tiles-w-lightning-timm-models-train.ipynb`
  - Tile-based training workflow using PyTorch Lightning and timm-based backbones (experiment notebook).

- `cancer-subtype-lightning-torch-inference-tiles.ipynb`
  - Tile-based inference workflow using PyTorch Lightning (experiment notebook).

### Python modules

- `download.py`
  - Downloads selected training images and thumbnails from the Kaggle competition using `kaggle competitions download`.
  - Expects an `updated_image_ids.json` file listing image ids and downloads both thumbnails and full images. :contentReference[oaicite:1]{index=1}

- `tiles.py`
  - Implements a tiling pipeline using `pyvips` to crop large images into fixed-size tiles, optionally subsampling tiles, filtering mostly-white tiles, and resizing tiles for downstream training/inference.
  - Includes a small dataset helper (`TilesFolderDataset`) for iterating over tiles on disk. :contentReference[oaicite:2]{index=2}

- `model.py`
  - Contains baseline PyTorch model code and supporting transforms/datasets, including:
    - A simple CNN architecture (`UBCOCEANNet1`)
    - Utility transforms (JPEG compression, aspect-ratio rescale, ordinal encoding)
    - Dataset wrappers for loading images from disk :contentReference[oaicite:3]{index=3}

## How the pipeline fits together (high level)

1. Download images (optional, if not running inside Kaggle with mounted inputs)
2. Generate tiles per slide image (tiling + pruning/filtering)
3. Train a model using tile datasets (Lightning-based notebooks or baseline training notebook)
4. Run inference (tile-based inference notebook), then aggregate/publish predictions

## Requirements (practical notes)

- Kaggle CLI configured (only needed if you use `download.py`)
- Python packages used across the repo commonly include:
  - torch, torchvision, pytorch-lightning
  - timm (for the timm-based training notebook)
  - numpy, pandas, scikit-learn
  - pyvips + libvips (for efficient large-image tiling)
  - Pillow, matplotlib

For the tiling pipeline, you may need system-level libvips installed for `pyvips` to work.

## Notes

This repo is shared as an example of batch ML workflow design for data-intensive image processing. The emphasis is on preprocessing strategy, execution constraints (memory/I/O), and workflow structure rather than on producing a polished, production-ready package.
