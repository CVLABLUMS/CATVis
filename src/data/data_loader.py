"""
Data loading utilities for CATVis research project.
Centralizes all data loading and preprocessing logic.
"""

import os
import re
import yaml
import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class CATVisDataLoader:
    """
    Centralized data loader for all CATVis components.
    Handles loading of EEG data, splits, captions, and mappings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = config['data']['root_dir']
        
        # Initialize data containers
        self.raw_data = None
        self.splits = None
        self.captions_data = None
        self.labels = None
        
        # Initialize mappings
        self.label_to_class = {}
        self.label_to_simple_class = {}
        self.image_to_class = {}
        self.image_to_simple_class = {}
        self.image_to_name = {}
        self.image_to_path = {}
        
        # Load all data
        self._load_all_data()
        
    def _load_all_data(self):
        """Load all required data files and generate mappings."""
        self._load_raw_data()
        self._load_captions()
        self._generate_mappings()
        
    def _load_raw_data(self):
        """Load EEG data and splits."""
        # Load EEG data
        eeg_path = os.path.join(self.data_dir, self.config['data']['eeg_file'])
        self.raw_data = torch.load(eeg_path, weights_only=True)
        
        # Load splits
        splits_path = os.path.join(self.data_dir, self.config['data']['splits_file'])
        self.splits = torch.load(splits_path, weights_only=True)
        
        self.labels = self.raw_data["labels"]
        print(f"Loaded EEG data with {len(self.labels)} classes")
        
    def _load_captions(self):
        """Load and process caption data."""
        captions_path = os.path.join(self.data_dir, self.config['data']['captions_file'])
        captions_with_bbox_data = torch.load(captions_path, weights_only=True)
        
        # Process captions data
        self.captions_data = {"captions": [], "bbox_labels": [], "images": []}
        for i in range(len(captions_with_bbox_data['images'])):
            caption = captions_with_bbox_data['captions_with_bbox'][i]['<CAPTION>']
            bbox_label = captions_with_bbox_data['captions_with_bbox'][i]['<CAPTION_TO_PHRASE_GROUNDING>']['labels']
            image = captions_with_bbox_data['images'][i]
            
            self.captions_data["captions"].append(caption)
            self.captions_data["bbox_labels"].append(bbox_label)
            self.captions_data["images"].append(image)
            
        print(f"Loaded {len(self.captions_data['captions'])} captions")
        
    def _generate_mappings(self):
        """Generate all the label and image mappings."""
        # Classes used in the paper
        classes_in_paper = [
            "dog", "cat", "butterfly", "sorrel", "capuchin", "elephant", "panda", "fish", 
            "airliner", "broom", "canoe", "phone", "mug", "convertible", "computer", "watch", 
            "guitar", "locomotive", "espresso", "chair", "golf", "piano", "iron", "jack", 
            "mailbag", "missile", "mitten", "bike", "tent", "pajama", "parachute", "pool", 
            "radio", "camera", "gun", "shoe", "banana", "pizza", "daisy", "bolete"
        ]
        
        # Load ImageNet class labels
        labels_path = os.path.join(self.data_dir, self.config['data']['imagenet_labels'])
        with open(labels_path, "r") as f:
            lines = f.read().split("\n")
            
        # 1K labels to all possible class values
        label_to_imagenet_classes = {}
        for line in lines:
            s = line.split(': ')
            try:
                label_to_imagenet_classes[s[0]] = s[1]
            except:
                label_to_imagenet_classes[s[0]] = None
        
        # Generate label mappings
        for idx, label in enumerate(self.labels):
            imagenet_class = label_to_imagenet_classes[label]
            
            # Use custom prompts if available, otherwise use imagenet class
            if label in self.config['class_prompts']:
                self.label_to_class[idx] = self.config['class_prompts'][label]
            else:
                self.label_to_class[idx] = imagenet_class
            
            # Generate simplified class names for paper
            s = re.split(r"[ ,!;.-]+", imagenet_class)
            simplified_class = list(set([c for c in s if c in classes_in_paper]))
            
            if len(simplified_class) == 0:
                print("No possible class found for: ", imagenet_class)
                print(s)
            elif len(simplified_class) > 1:
                print("Multiple possible classes for: ", imagenet_class)
            else:
                self.label_to_simple_class[idx] = simplified_class[0]
        
        # Special case for jack-o-lantern -> pumpkin
        if 'n03590841' in self.labels:
            jack_idx = self.labels.index('n03590841')
            self.label_to_simple_class[jack_idx] = "pumpkin"
        
        # Generate image mappings
        for i, image_name in enumerate(self.raw_data['images']):
            label_idx = self.labels.index(image_name.split('_')[0])
            self.image_to_class[i] = self.label_to_class[label_idx]
            self.image_to_simple_class[i] = self.label_to_simple_class[label_idx]
            self.image_to_name[i] = image_name
            self.image_to_path[i] = image_name.split('_')[0] + "/" + image_name + ".JPEG"
        
        # Verify mappings
        assert label_to_imagenet_classes[self.labels[10]] == "canoe"
        assert self.label_to_simple_class[10] == "canoe"
        
        # Verify image paths exist
        missing_paths = []
        for image_path in self.image_to_path.values():
            full_path = os.path.join(self.data_dir, self.config['data']['imagenet_images'], image_path)
            if not os.path.exists(full_path):
                missing_paths.append(image_path)
        
        if missing_paths:
            print(f"Warning: {len(missing_paths)} image paths don't exist")
            print("First few missing paths:", missing_paths[:5])
            
        print("Generated all mappings successfully")
        
    def get_dataset_dataframe(self) -> pd.DataFrame:
        """Create the main dataset DataFrame."""
        captions_df = pd.DataFrame(self.captions_data)
        dataset_df = pd.DataFrame(self.raw_data["dataset"])
        dataset_df["class"] = dataset_df["label"].map(self.label_to_class)
        
        # Join with captions
        df = dataset_df.join(
            captions_df[["captions", "bbox_labels"]], 
            on="image"
        )[["subject", "eeg", "captions", "bbox_labels", "class", "label", "image"]]
        
        # Handle duplicates in bbox_labels
        df["bbox_labels"] = df["bbox_labels"].apply(
            lambda labels: list(dict.fromkeys(labels))
        )
        
        return df
        
    def get_train_val_test_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get train, validation, and test splits."""
        train_split = self.splits["splits"][0]["train"]
        val_split = self.splits["splits"][0]["val"]
        test_split = self.splits["splits"][0]["test"]
        
        train_df = df.iloc[train_split]
        val_df = df.iloc[val_split]
        test_df = df.iloc[test_split]
        
        return train_df, val_df, test_df
        
    def prepare_eeg_data(self, df: pd.DataFrame, time_low: int = 20, time_high: int = 460) -> Tuple[np.ndarray, List]:
        """Prepare EEG data by extracting time windows."""
        eeg_data = np.array(df['eeg'].apply(lambda x: x[:, time_low:time_high]).tolist())
        labels = df['label'].tolist()
        return eeg_data, labels 