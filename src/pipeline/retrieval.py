"""
Text Retrieval Module for CATVis Image Generation Pipeline.
Extracted from original catvis_pipeline.py notebook.
"""

import torch
import torch.nn.functional as F
import clip
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import EEGTextDataset


class TextRetrieval:
    """
    Text retrieval system for the CATVis image generation pipeline.
    Handles EEG->text retrieval using the trained contrastive model.
    """
    
    def __init__(self, config: Dict[str, Any], contrastive_model, device: torch.device):
        self.config = config
        self.device = device
        self.contrastive_model = contrastive_model
        
        # Load CLIP for text encoding
        clip_model_name = config['contrastive_training']['clip_model']
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
            
        # Retrieval corpus - will be set during setup
        self.unique_caption_embeds = None
        self.unique_captions = None
        
    def setup_retrieval_corpus(self, test_df):
        """
        Set up the retrieval corpus with unique captions.
        Replicates the corpus creation from original pipeline.
        """
        print("Setting up retrieval corpus...")
        
        # Project unique captions into CLIP space
        retrieval_df = test_df.drop_duplicates(subset=["captions"])
        retrieval_dataset = EEGTextDataset(retrieval_df)
        retrieval_dataloader = DataLoader(retrieval_dataset, batch_size=128, shuffle=False)
        
        self.unique_caption_embeds, self.unique_captions = self._extract_embeddings(retrieval_dataloader)
        self.unique_caption_embeds = self.unique_caption_embeds.to(self.device)
        
        print(f"Retrieval corpus ready with {len(self.unique_captions)} unique captions")
        
    def _extract_embeddings(self, dataloader) -> Tuple[torch.Tensor, List[str]]:
        """Extract text embeddings using CLIP (as in original pipeline)."""
        all_text_embeds = []
        all_text_labels = []

        with torch.no_grad():
            for eeg_batch, text_batch in tqdm(dataloader, desc="Extracting embeddings"):
                text_tokens = clip.tokenize(text_batch, truncate=True).to(self.device)
                text_emb = self.clip_model.encode_text(text_tokens).float()
                text_emb = F.normalize(text_emb, dim=-1)

                all_text_embeds.append(text_emb.cpu())
                all_text_labels.extend(text_batch)

        all_text_embeds = torch.cat(all_text_embeds, dim=0)  # [N, 768]
        return all_text_embeds, all_text_labels
    
    def retrieve_top_k_from_eeg(self, eeg_tensor: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k captions for given EEG tensor.
        Replicates the retrieval function from original pipeline.
        
        Args:
            eeg_tensor: EEG tensor of shape [1, n_chans, n_times] or [n_chans, n_times]
            k: Number of top captions to retrieve
            
        Returns:
            List of (caption, score) tuples
        """
        if len(eeg_tensor.shape) == 2:
            eeg_tensor = eeg_tensor.unsqueeze(0)  # Add batch dimension
            
        eeg_tensor = eeg_tensor.to(self.device)
        
        self.contrastive_model.eval()
        with torch.no_grad():
            emb = self.contrastive_model(eeg_tensor)  # [1, 768]
            emb = F.normalize(emb, dim=-1)

        # Compute similarities
        sim = emb @ self.unique_caption_embeds.t()  # [1, N]
        sim = sim.squeeze(0)  # [N]

        # Get top-k results
        sorted_idx = torch.argsort(sim, descending=True)
        top_indices = sorted_idx[:k]
        
        results = []
        for idx_ in top_indices:
            caption = self.unique_captions[idx_]
            score = sim[idx_].item()
            results.append((caption, score))
            
        return results
    
    # Note: Retrieval evaluation now handled by shared function in evaluation.metrics
    # to eliminate code duplication and ensure consistency 