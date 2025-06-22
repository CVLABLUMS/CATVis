"""
EEG Classification model for CATVis.
"""

import torch
import torch.nn as nn
from braindecode.models import EEGConformer
from typing import Dict, Any, Tuple


class EEGClassifier(nn.Module):
    """
    EEG Classifier based on EEGConformer from braindecode.
    Modified to produce CLIP-aligned 768-dimensional features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(EEGClassifier, self).__init__()
        self.config = config
        
        # Extract model parameters from config
        model_config = config['eeg_classification']
        model_params = model_config['model_params']
        
        # Create base EEGConformer model
        self.model = EEGConformer(
            n_outputs=model_config['n_outputs'],
            n_chans=model_config['n_chans'],
            n_times=model_config['n_times'],
            n_filters_time=model_params['n_filters_time'],
            filter_time_length=model_params['filter_time_length'],
            pool_time_length=model_params['pool_time_length'],
            pool_time_stride=model_params['pool_time_stride'],
            final_fc_length=model_params['final_fc_length'],
            add_log_softmax=False,
            return_features=True
        )
        
        # Modify final layers to produce 768-dimensional features (CLIP-aligned)
        self.model.fc.fc[3] = nn.Linear(in_features=256, out_features=768, bias=True)
        self.model.final_layer.final_layer[0] = nn.Linear(in_features=768, out_features=model_config['n_outputs'], bias=True)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input EEG tensor of shape [batch_size, n_chans, n_times]
            
        Returns:
            Tuple of (classification_outputs, eeg_embeddings)
            - classification_outputs: [batch_size, n_classes] 
            - eeg_embeddings: [batch_size, 768] feature embeddings
        """
        outputs, embeddings = self.model(x)
        return outputs, embeddings
        
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get only the 768-dimensional embeddings.
        
        Args:
            x: Input EEG tensor of shape [batch_size, n_chans, n_times]
            
        Returns:
            eeg_embeddings: [batch_size, 768] feature embeddings
        """
        _, embeddings = self.forward(x)
        return embeddings
        
    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {checkpoint_path}")
        
    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}") 