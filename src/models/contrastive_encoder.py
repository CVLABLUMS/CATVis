"""
Contrastive EEG Encoder for CATVis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGConformer
from typing import Dict, Any, Tuple


class ContrastiveEncoder(nn.Module):
    """
    EEG Encoder for contrastive learning with text embeddings.
    Based on EEGConformer, modified to output 768-dimensional embeddings
    aligned with CLIP text embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(ContrastiveEncoder, self).__init__()
        self.config = config
        
        # Extract model parameters from config
        model_config = config['eeg_classification']  # Uses same architecture as classifier
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
        
        # Remove final classification layer, leaving us with 768-d embeddings
        self.model.final_layer = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input EEG tensor of shape [batch_size, n_chans, n_times]
            
        Returns:
            eeg_embeddings: [batch_size, 768] feature embeddings
        """
        embeddings = self.model(x)
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


def clip_style_contrastive_loss(eeg_embeds: torch.Tensor, text_embeds: torch.Tensor, 
                               temperature: float = 0.07) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    CLIP-style contrastive loss for EEG-text alignment.
    
    Args:
        eeg_embeds: [batch_size, embed_dim] EEG embeddings
        text_embeds: [batch_size, embed_dim] Text embeddings
        temperature: Temperature parameter for contrastive loss
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    batch_size = eeg_embeds.size(0)

    # Similarity matrices
    logits_eeg = eeg_embeds @ text_embeds.t() / temperature     # [B, B]
    logits_text = text_embeds @ eeg_embeds.t() / temperature    # [B, B]

    labels = torch.arange(batch_size, device=eeg_embeds.device) # [0..B-1]

    # InfoNCE loss (two-way)
    loss_eeg = F.cross_entropy(logits_eeg, labels)
    loss_text = F.cross_entropy(logits_text, labels)
    loss = (loss_eeg + loss_text) / 2.0

    # Accuracy metrics (in-batch retrieval)
    acc_eeg = (logits_eeg.argmax(dim=-1) == labels).float().mean()
    acc_text = (logits_text.argmax(dim=-1) == labels).float().mean()

    metrics = {
        "loss": loss.item(),
        "acc_eeg": acc_eeg.item(),
        "acc_text": acc_text.item(),
    }
    return loss, metrics 