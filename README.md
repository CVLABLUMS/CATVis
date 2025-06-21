# CATVis: Context-Aware Thought Visualization

This repository contains the implementation of the CATVis (Context-Aware Thought Visualization) research project for generating images from EEG brain signals.

## Repository Structure

```
CATVis/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   └── README.md (instructions for data placement)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── eeg_classifier.py
│   │   └── contrastive_encoder.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_classifier.py
│   │   └── train_contrastive.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── image_generator.py
│   │   └── retrieval.py
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
├── scripts/
│   ├── train_eeg_classifier.py
│   ├── train_contrastive_model.py
│   ├── run_pipeline.py
│   └── evaluate_results.py
└── checkpoints/
    └── README.md
```

## Pipeline Overview

1. **EEG Classification**: Train EEGConformer model for 40-class classification
2. **Contrastive Alignment**: Train EEG-text alignment using CLIP embeddings  
3. **Image Generation**: Use trained models in Stable Diffusion pipeline
4. **Evaluation**: Compute metrics (FID, IS, Classification Accuracy)

## Setup Instructions

1. Install dependencies: `pip install -r requirements.txt`
2. Place data files in `data/` directory (see data/README.md)
3. Run training scripts in order
4. Execute pipeline and evaluation

## Usage

### 1. Train Models

```bash
# Train EEG classifier (required first)
python scripts/train_eeg_classifier.py

# Train contrastive model (required second)
python scripts/train_contrastive_model.py
```

### 2. Run Image Generation Pipeline

```bash
# Generate images for all subjects
python scripts/run_pipeline.py

# Generate for specific subjects only
python scripts/run_pipeline.py --subjects "1,2,4"

# Test run with limited batches
python scripts/run_pipeline.py --max-batches 5

# Dry run (setup only, no generation)
python scripts/run_pipeline.py --dry-run
```

### 3. Evaluate Results

```bash
# Run all evaluation metrics
python scripts/evaluate_results.py

# Evaluate specific results directory
python scripts/evaluate_results.py --results-dir ./custom_results

# Run only generation metrics (skip classification)
python scripts/evaluate_results.py --metrics generation
```

### Custom Configuration

All scripts accept a `--config` parameter to specify custom configuration:

```bash
python scripts/train_eeg_classifier.py --config custom_config.yaml
```

## Research Paper

[Link to paper when published] 