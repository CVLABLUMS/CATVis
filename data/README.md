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

- [EEG visual classification dataset](https://tinyurl.com/eeg-visual-classification): Download `eeg_55_95_std.pth` and `block_splits_by_image_all.pth`, and place in this directory.
- ImageNet-40 subset (Kaggle: [imagenet-40](https://www.kaggle.com/datasets/tariq9mehmood9/imagenet-40)) 

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