# CATVis: EEG-to-Image Generation Pipeline

This repository contains the implementation of the CATVis (Category-Aligned Thought Visualization) research project for generating images from EEG brain signals.

## Repository Structure

```
catvis-refactoring/
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

## Research Paper

[Link to paper when published] 