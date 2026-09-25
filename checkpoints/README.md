# Model Checkpoints Directory

This directory stores trained model weights and checkpoints.

Pretrained `eeg_classifier_best.pth` and `contrastive_model_best.pth` checkpoints are included, so you can skip training and run `--test-only` or the generation pipeline directly. Re-running training overwrites them.

## Files Generated During Training

### EEG Classification Model
- `eeg_classifier_best.pth` - Best EEG classifier model weights
- `eeg_classifier_latest.pth` - Latest checkpoint (optional)

### Contrastive Alignment Model  
- `contrastive_model_best.pth` - Best contrastive alignment model weights
- `contrastive_model_latest.pth` - Latest checkpoint (optional)

## Usage

The trained models are automatically loaded by the pipeline:
- Classification model: Used for EEG -> class prediction
- Contrastive model: Used for EEG -> text retrieval

## Model Architecture Details

### EEG Classifier
- Base: EEGConformer from braindecode
- Output: 40 classes (ImageNet subset)
- Features: 768-dimensional embeddings (CLIP-aligned)

### Contrastive Model
- Base: EEGConformer modified for contrastive learning
- Output: 768-dimensional embeddings
- Training: Contrastive loss with CLIP text embeddings

## File Sizes

Expect checkpoint files to be approximately:
- EEG classifier: ~3 MB
- Contrastive model: ~3 MB