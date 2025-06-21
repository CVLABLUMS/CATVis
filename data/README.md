# Data Directory Setup

This directory should contain the required data files for the CATVis (Context-Aware Thought Visualization) research project.

## Required Files

Place the following files in this directory:

### Core Data Files
- `eeg_55_95_std.pth` - EEG data tensor file
- `block_splits_by_image_all.pth` - Train/validation/test splits 
- `captions_with_bbox_data.pth` - Image captions with bounding box data
- `imagenet_class_labels.txt` - ImageNet class label mappings

### Image Data
- `imageNet_images/` - Directory containing ImageNet images organized by class
  - Structure: `imageNet_images/{class_id}/{image_name}.JPEG`
  - Example: `imageNet_images/n02106662/n02106662_6706.JPEG`

## Data Sources

The original data comes from:
- EEG visual classification dataset (Kaggle: eeg-visual-classification-new)
- ImageNet-40 subset (Kaggle: imagenet-40)

## Directory Structure After Setup

```
data/
├── README.md
├── eeg_55_95_std.pth
├── block_splits_by_image_all.pth
├── captions_with_bbox_data.pth
├── imagenet_class_labels.txt
└── imageNet_images/
    ├── n02106662/
    │   ├── n02106662_6706.JPEG
    │   └── ...
    ├── n02124075/
    │   └── ...
    └── ...
```

## Notes

- The code expects these exact file names and structure
- Make sure all image files are accessible and not corrupted
- Total dataset contains 40 ImageNet classes with EEG recordings
- Missing files will cause the pipeline to fail with clear error messages 