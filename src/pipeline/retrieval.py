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
    
    def evaluate_retrieval_accuracy(self, test_df) -> Dict[str, float]:
        """
        Evaluate retrieval accuracy on test set.
        Replicates the evaluation from original pipeline.
        """
        print("Evaluating retrieval accuracy...")
        
        # Create test dataset
        test_dataset = EEGTextDataset(test_df)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Extract all EEG embeddings and true captions
        all_eeg_embeds = []
        all_true_captions = []

        with torch.no_grad():
            for eeg_batch, caption_batch in tqdm(test_loader, desc="Extracting test EEG"):
                eeg_batch = eeg_batch.to(self.device)
                eeg_embeds = self.contrastive_model(eeg_batch)  # [B, 768]
                eeg_embeds = F.normalize(eeg_embeds, dim=-1)
                all_eeg_embeds.append(eeg_embeds.cpu())
                all_true_captions.extend(caption_batch)

        all_eeg_embeds = torch.cat(all_eeg_embeds, dim=0)  # [N, 768]
        N = all_eeg_embeds.size(0)

        # Move to device for computation
        all_eeg_embeds = all_eeg_embeds.to(self.device)
        sim_e2t = all_eeg_embeds @ self.unique_caption_embeds.t()  # [N, M]

        # Create caption to index mapping
        caption_to_idx = {cap: idx for idx, cap in enumerate(self.unique_captions)}

        # Get true text indices
        true_text_indices = []
        for cap in all_true_captions:
            correct_idx = caption_to_idx[cap]
            true_text_indices.append(correct_idx)
        true_text_indices = torch.tensor(true_text_indices, device=self.device)

        # Compute recall metrics
        topks = [1, 5, 10]
        hits_e2t = {k: 0 for k in topks}

        for i in range(N):
            row = sim_e2t[i]  # [M]
            sorted_idx = torch.argsort(row, descending=True)
            correct_idx = true_text_indices[i]
            rank = (sorted_idx == correct_idx).nonzero(as_tuple=True)[0].item()
            for k in topks:
                if rank < k:
                    hits_e2t[k] += 1

        # Calculate and return results
        results = {}
        print("\n===== RETRIEVAL EVALUATION (EEG->Text) =====")
        for k in topks:
            recall = hits_e2t[k] / N * 100
            results[f'recall@{k}'] = recall
            print(f"Recall@{k}: {recall:.2f}%")
            
        return results 