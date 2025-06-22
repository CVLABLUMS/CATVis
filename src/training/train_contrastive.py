"""
Contrastive EEG-Text Training Module for CATVis.
Extracted from original cross_modal_alignment.py notebook.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import clip
from tqdm import tqdm
from typing import Dict, Any, Tuple, List

from models import ContrastiveEncoder, clip_style_contrastive_loss
from data import CATVisDataLoader, DataPreprocessor


class ContrastiveTrainer:
    """
    Trainer class for contrastive EEG-text alignment.
    Extracted from original research code with exact training methodology preserved.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # Training configuration
        self.num_epochs = config['contrastive_training']['num_epochs']
        self.learning_rate = config['contrastive_training']['learning_rate']
        self.batch_size = config['contrastive_training']['batch_size']
        self.temperature = config['contrastive_training']['temperature']
        self.patience = config['contrastive_training']['patience']
        
        # Initialize EEG model
        self.eeg_model = ContrastiveEncoder(config).to(device)
        
        # Initialize CLIP model (frozen)
        clip_model_name = config['contrastive_training']['clip_model']
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        # Optimizer (only for EEG model)
        self.optimizer = torch.optim.Adam(self.eeg_model.parameters(), lr=self.learning_rate)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        # Checkpoint paths
        self.checkpoint_dir = config['checkpoints']['root_dir']
        self.model_save_path = os.path.join(self.checkpoint_dir, config['checkpoints']['contrastive_model'])
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.eeg_model.train()
        train_losses = []
        train_accs_eeg = []
        train_accs_text = []

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for eeg_batch, text_batch in train_pbar:
            eeg_batch = eeg_batch.to(self.device)

            # Encode text (frozen CLIP)
            with torch.no_grad():
                text_tokens = clip.tokenize(text_batch, truncate=True).to(self.device)
                text_embeds = self.clip_model.encode_text(text_tokens).float()
                text_embeds = F.normalize(text_embeds, dim=-1)

            # Encode EEG (trainable)
            eeg_embeds = self.eeg_model(eeg_batch)
            eeg_embeds = F.normalize(eeg_embeds, dim=-1)

            # Contrastive loss
            loss, metrics = clip_style_contrastive_loss(eeg_embeds, text_embeds, self.temperature)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update train metrics
            train_losses.append(loss.item())
            train_accs_eeg.append(metrics["acc_eeg"])
            train_accs_text.append(metrics["acc_text"])

            # Show info in progress bar
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc_eeg": f"{metrics['acc_eeg']:.4f}",
                "acc_text": f"{metrics['acc_text']:.4f}"
            })

        return {
            'loss': np.mean(train_losses),
            'acc_eeg': np.mean(train_accs_eeg),
            'acc_text': np.mean(train_accs_text)
        }
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.eeg_model.eval()
        val_losses = []
        val_accs_eeg = []
        val_accs_text = []

        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for eeg_batch, text_batch in val_pbar:
                eeg_batch = eeg_batch.to(self.device)

                text_tokens = clip.tokenize(text_batch, truncate=True).to(self.device)
                text_embeds = self.clip_model.encode_text(text_tokens).float()
                text_embeds = F.normalize(text_embeds, dim=-1)

                eeg_embeds = self.eeg_model(eeg_batch)
                eeg_embeds = F.normalize(eeg_embeds, dim=-1)

                val_loss, val_metrics = clip_style_contrastive_loss(eeg_embeds, text_embeds, self.temperature)
                val_losses.append(val_loss.item())
                val_accs_eeg.append(val_metrics["acc_eeg"])
                val_accs_text.append(val_metrics["acc_text"])

        return {
            'loss': np.mean(val_losses),
            'acc_eeg': np.mean(val_accs_eeg),
            'acc_text': np.mean(val_accs_text)
        }
    
    def train(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """
        Complete training loop with early stopping.
        Preserves exact training methodology from original notebook.
        """
        print(f"Starting contrastive training for {self.num_epochs} epochs...")
        
        # Training history
        train_history = {'loss': [], 'acc_eeg': [], 'acc_text': []}
        val_history = {'loss': [], 'acc_eeg': [], 'acc_text': []}
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            for key, value in train_metrics.items():
                train_history[key].append(value)

            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            for key, value in val_metrics.items():
                val_history[key].append(value)

            # Print epoch summary
            print(f"[Epoch {epoch+1}/{self.num_epochs}] "
                  f"TrainLoss: {train_metrics['loss']:.4f}, ValLoss: {val_metrics['loss']:.4f} | "
                  f"TrainAcc_EEG: {train_metrics['acc_eeg']:.4f}, ValAcc_EEG: {val_metrics['acc_eeg']:.4f} | "
                  f"TrainAcc_Text: {train_metrics['acc_text']:.4f}, ValAcc_Text: {val_metrics['acc_text']:.4f}")

            # Early stopping & checkpointing
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.early_stop_counter = 0
                self.eeg_model.save_checkpoint(self.model_save_path)
                print(f"  --> Saved new best model at epoch {epoch+1} (val_loss={self.best_val_loss:.4f})")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

        print(f"Training complete. Best val_loss={self.best_val_loss:.4f}")
        
        # Load best model
        self.eeg_model.load_pretrained_weights(self.model_save_path)
        
        return {
            'train': train_history,
            'val': val_history
        }
    
    def evaluate_retrieval(self, test_loader) -> Dict[str, float]:
        """
        Evaluate EEG->Text retrieval performance.
        Replicates evaluation from original notebook.
        """
        print("Evaluating retrieval performance...")
        
        # Extract unique captions for retrieval corpus
        retrieval_df = test_loader.dataset.df.drop_duplicates(subset=["captions"])
        from data import EEGTextDataset
        from torch.utils.data import DataLoader
        retrieval_dataset = EEGTextDataset(retrieval_df)
        retrieval_dataloader = DataLoader(retrieval_dataset, batch_size=128, shuffle=False)
        
        # Get unique caption embeddings
        unique_caption_embeds, unique_captions = self._extract_text_embeddings(retrieval_dataloader)
        
        # Get all EEG embeddings and true captions
        all_eeg_embeds, all_true_captions = self._extract_eeg_embeddings(test_loader)
        
        # Compute similarities
        all_eeg_embeds = all_eeg_embeds.to(self.device)
        unique_caption_embeds = unique_caption_embeds.to(self.device)
        
        sim_e2t = all_eeg_embeds @ unique_caption_embeds.t()  # [N, M]
        
        # Create caption to index mapping
        caption_to_idx = {cap: idx for idx, cap in enumerate(unique_captions)}
        
        # Get true text indices
        true_text_indices = []
        for cap in all_true_captions:
            correct_idx = caption_to_idx[cap]
            true_text_indices.append(correct_idx)
        true_text_indices = torch.tensor(true_text_indices, device=self.device)
        
        # Compute retrieval metrics
        N = len(all_true_captions)
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
        
        # Calculate recall scores
        results = {}
        print("\n===== RETRIEVAL EVALUATION (EEG->Text) =====")
        for k in topks:
            recall = hits_e2t[k] / N * 100
            results[f'recall@{k}'] = recall
            print(f"Recall@{k}: {recall:.2f}%")
        
        return results
    
    def _extract_text_embeddings(self, dataloader) -> Tuple[torch.Tensor, List[str]]:
        """Extract text embeddings using CLIP."""
        all_text_embeds = []
        all_text_labels = []

        self.eeg_model.eval()
        with torch.no_grad():
            for eeg_batch, text_batch in tqdm(dataloader, desc="Extracting text embeddings"):
                text_tokens = clip.tokenize(text_batch, truncate=True).to(self.device)
                text_emb = self.clip_model.encode_text(text_tokens).float()
                text_emb = F.normalize(text_emb, dim=-1)

                all_text_embeds.append(text_emb.cpu())
                all_text_labels.extend(text_batch)

        all_text_embeds = torch.cat(all_text_embeds, dim=0)  # [N, 768]
        return all_text_embeds, all_text_labels
    
    def _extract_eeg_embeddings(self, dataloader) -> Tuple[torch.Tensor, List[str]]:
        """Extract EEG embeddings."""
        all_eeg_embeds = []
        all_true_captions = []

        self.eeg_model.eval()
        with torch.no_grad():
            for eeg_batch, caption_batch in tqdm(dataloader, desc="Extracting EEG embeddings"):
                eeg_batch = eeg_batch.to(self.device)
                eeg_embeds = self.eeg_model(eeg_batch)  # [B, 768]
                eeg_embeds = F.normalize(eeg_embeds, dim=-1)
                all_eeg_embeds.append(eeg_embeds.cpu())
                all_true_captions.extend(caption_batch)

        all_eeg_embeds = torch.cat(all_eeg_embeds, dim=0)  # [N, 768]
        return all_eeg_embeds, all_true_captions


def train_contrastive_model(config_path: str = "config/config.yaml",
                          test_only: bool = False,
                          checkpoint_path: str = None) -> ContrastiveTrainer:
    """
    Complete training pipeline for contrastive EEG-text alignment.
    Replicates the exact workflow from original cross_modal_alignment.py notebook.
    """
    from data import load_config
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_loader = CATVisDataLoader(config)
    df = data_loader.get_dataset_dataframe()
    train_df, val_df, test_df = data_loader.get_train_val_test_splits(df)
    
    print(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets and data loaders for contrastive learning
    preprocessor = DataPreprocessor(config)
    train_dataset, val_dataset, test_dataset = preprocessor.create_contrastive_datasets(
        train_df, val_df, test_df
    )
    
    # Use contrastive batch size
    batch_size = config['contrastive_training']['batch_size']
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )
    
    # Initialize trainer
    trainer = ContrastiveTrainer(config, device)
    
    if test_only:
        # Test-only mode: load existing checkpoint and evaluate
        checkpoint_to_load = checkpoint_path if checkpoint_path else trainer.model_save_path
        
        if not os.path.exists(checkpoint_to_load):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_to_load}")
        
        print(f"Loading checkpoint: {checkpoint_to_load}")
        trainer.eeg_model.load_pretrained_weights(checkpoint_to_load)
        
    else:
        # Training mode (default)
        print("Starting contrastive training...")
        training_history = trainer.train(train_loader, val_loader)
    
    # Evaluate retrieval performance (common for both modes)
    retrieval_results = trainer.evaluate_retrieval(test_loader)
    
    return trainer 