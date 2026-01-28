## Overview

This project explores the design of batch-oriented ML pipelines for data-intensive image processing workloads.
It focuses on preprocessing, staged execution, GPU-backed training/evaluation, and operational constraints
(memory, I/O throughput, and reproducibility).

## What this demonstrates

- Batch-style preprocessing of multi-gigabyte image artifacts via tiling
- GPU-backed training and evaluation workflows
- Managing memory and data-loading constraints
- Reproducible experiment execution in a cloud environment