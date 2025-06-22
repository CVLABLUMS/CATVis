"""
Data preprocessing and PyTorch dataset classes for CATVis.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, Any, List, Tuple


class EEGDataset(Dataset):
    """Dataset for EEG data with labels."""
    
    def __init__(self, eeg_data: np.ndarray, labels: List[int], images: List[int] = None, 
                 subjects: List[int] = None, captions: List[str] = None):
        self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.images = torch.tensor(images, dtype=torch.long) if images is not None else None
        self.subjects = torch.tensor(subjects, dtype=torch.long) if subjects is not None else None
        self.captions = captions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'eeg': self.eeg_data[idx],
            'label': self.labels[idx]
        }
        
        if self.images is not None:
            sample['image'] = self.images[idx]
        if self.subjects is not None:
            sample['subject'] = self.subjects[idx]
        if self.captions is not None:
            sample['caption'] = self.captions[idx]
            
        return sample


class EEGTextDataset(Dataset):
    """Dataset for EEG-text pairs used in contrastive learning."""
    
    def __init__(self, df, tl: int = 20, th: int = 460):
        """
        Args:
            df: DataFrame with EEG data and captions
            tl: Time low (start of time window)
            th: Time high (end of time window)
        """
        self.df = df.reset_index(drop=True)
        self.tl = tl
        self.th = th

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_tensor = row["eeg"]  # shape [n_chans, n_times]
        eeg_tensor = eeg_tensor[:, self.tl:self.th]  # shape: [128, 440]
        caption = row["captions"]
        
        return eeg_tensor, caption


class DataPreprocessor:
    """
    Data preprocessing utilities for CATVis.
    Handles creation of PyTorch datasets and data loaders.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def create_classification_datasets(self, train_df, val_df, test_df, 
                                     time_low: int = None, time_high: int = None) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        """Create datasets for EEG classification using TensorDataset (like original notebook)."""
        if time_low is None:
            time_low = self.config['eeg_classification']['time_low']
        if time_high is None:
            time_high = self.config['eeg_classification']['time_high']
            
        # Extract EEG data and labels (exactly like original notebook)
        eeg_train = np.array(train_df['eeg'].apply(lambda x: x[:, time_low:time_high]).tolist())
        train_labels = train_df['label'].tolist()
        
        eeg_val = np.array(val_df['eeg'].apply(lambda x: x[:, time_low:time_high]).tolist())
        val_labels = val_df['label'].tolist()
        
        eeg_test = np.array(test_df['eeg'].apply(lambda x: x[:, time_low:time_high]).tolist())
        test_labels = test_df['label'].tolist()
        
        # Create TensorDatasets (exactly like original notebook)
        train_dataset = TensorDataset(torch.tensor(eeg_train, dtype=torch.float32),
                                      torch.tensor(train_labels, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(eeg_val, dtype=torch.float32),
                                    torch.tensor(val_labels, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(eeg_test, dtype=torch.float32),
                                     torch.tensor(test_labels, dtype=torch.long))
        
        return train_dataset, val_dataset, test_dataset
        
    def create_contrastive_datasets(self, train_df, val_df, test_df, 
                                  time_low: int = None, time_high: int = None) -> Tuple[EEGTextDataset, EEGTextDataset, EEGTextDataset]:
        """Create datasets for contrastive learning."""
        if time_low is None:
            time_low = self.config['eeg_classification']['time_low']
        if time_high is None:
            time_high = self.config['eeg_classification']['time_high']
            
        train_dataset = EEGTextDataset(train_df, time_low, time_high)
        val_dataset = EEGTextDataset(val_df, time_low, time_high)
        test_dataset = EEGTextDataset(test_df, time_low, time_high)
        
        return train_dataset, val_dataset, test_dataset
        
    def create_pipeline_dataset(self, test_df, time_low: int = None, time_high: int = None) -> EEGDataset:
        """Create dataset for the generation pipeline."""
        if time_low is None:
            time_low = self.config['eeg_classification']['time_low']
        if time_high is None:
            time_high = self.config['eeg_classification']['time_high']
            
        eeg_test = np.array(test_df['eeg'].apply(lambda x: x[:, time_low:time_high]).tolist())
        test_labels = test_df['label'].tolist()
        test_images = test_df['image'].tolist()
        test_subjects = test_df['subject'].tolist()
        test_captions = test_df['captions'].tolist()
        
        return EEGDataset(eeg_test, test_labels, test_images, test_subjects, test_captions)
        
    def create_data_loaders(self, train_dataset, val_dataset, test_dataset, 
                          batch_size: int = None, shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch data loaders."""
        if batch_size is None:
            batch_size = self.config['eeg_classification']['batch_size']
            
        # Set up generator exactly like original notebook
        torch_generator = torch.Generator()
        torch_generator.manual_seed(45)  # Use same seed as original
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle_train,
            generator=torch_generator
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader 