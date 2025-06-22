"""
EEG Classification Training Module for CATVis.
Extracted from original eeg_classification.py notebook.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple, List

from models import EEGClassifier
from data import CATVisDataLoader, DataPreprocessor


class ClassifierTrainer:
    """
    Trainer class for EEG Classification model.
    Extracted from original research code with exact training methodology preserved.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # Training configuration
        self.num_epochs = config['eeg_classification']['num_epochs']
        self.learning_rate = config['eeg_classification']['learning_rate']
        self.batch_size = config['eeg_classification']['batch_size']
        
        # Initialize model
        self.model = EEGClassifier(config).to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training tracking
        self.loss_list = []
        self.accuracy_list = []
        self.val_loss_list = []
        self.val_accuracy_list = []
        self.best_val_acc = 0.0
        
        # Checkpoint path
        self.checkpoint_dir = config['checkpoints']['root_dir']
        self.model_save_path = os.path.join(self.checkpoint_dir, config['checkpoints']['eeg_classifier'])
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0.0
        
        for i, data in enumerate(train_loader):
            eeg, label = data  # Tuple unpacking like original notebook
            eeg, label = eeg.to(self.device), label.to(self.device)

            self.optimizer.zero_grad()
            outputs, eeg_embeddings = self.model(eeg)
            loss = self.criterion(outputs, label)
            loss.backward()
            self.optimizer.step()

            correct += (outputs.argmax(dim=-1) == label).float().sum().item()
            running_loss += loss.item()
        
        # Calculate average loss and accuracy for the epoch
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / len(train_loader.dataset)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0.0

        with torch.no_grad():
            for val_data in val_loader:
                eeg_val, label_val = val_data  # Tuple unpacking like original notebook
                eeg_val, label_val = eeg_val.to(self.device), label_val.to(self.device)

                outputs_val, eeg_embeddings_val = self.model(eeg_val)
                batch_val_loss = self.criterion(outputs_val, label_val)

                val_loss += batch_val_loss.item()
                val_correct += (outputs_val.argmax(dim=-1) == label_val).float().sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / len(val_loader.dataset)
        
        return avg_val_loss, val_accuracy
    
    def train(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """
        Complete training loop.
        Preserves exact training methodology from original notebook.
        """
        print(f"Starting EEG Classification training for {self.num_epochs} epochs...")
        
        for epoch in tqdm(range(self.num_epochs), desc="Training Progress"):
            # Training phase
            avg_loss, accuracy = self.train_epoch(train_loader)
            self.loss_list.append(avg_loss)
            self.accuracy_list.append(accuracy)

            # Validation phase
            avg_val_loss, val_accuracy = self.validate_epoch(val_loader)
            self.val_loss_list.append(avg_val_loss)
            self.val_accuracy_list.append(val_accuracy)

            # Print progress (every 10 epochs or last epoch)
            if (epoch + 1) % 10 == 0 or epoch == self.num_epochs - 1:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                      f'TrL: {avg_loss:.4f}, VaL: {avg_val_loss:.4f}, '
                      f'TrA: {accuracy:.2f}%, VaA: {val_accuracy:.2f}%')

            # Save best model
            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                self.model.save_checkpoint(self.model_save_path)
                print(f'Best model saved at epoch {epoch + 1} (Val Acc: {val_accuracy:.2f}%)')

        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Load best model
        self.model.load_pretrained_weights(self.model_save_path)
        
        return {
            'train_loss': self.loss_list,
            'train_accuracy': self.accuracy_list,
            'val_loss': self.val_loss_list,
            'val_accuracy': self.val_accuracy_list
        }
    
    def test(self, test_loader) -> Dict[str, Any]:
        """
        Test the trained model.
        Returns predictions and metrics as in original notebook.
        """
        print("Testing model...")
        
        self.model.eval()
        test_loss = 0.0
        test_correct = 0.0
        
        # Lists to store ground truth and predictions (for evaluation)
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for test_data in test_loader:
                eeg_test, label_test = test_data  # Tuple unpacking like original notebook
                eeg_test, label_test = eeg_test.to(self.device), label_test.to(self.device)

                outputs_test, eeg_embeddings_test = self.model(eeg_test)
                batch_test_loss = self.criterion(outputs_test, label_test)

                test_loss += batch_test_loss.item()
                test_correct += (outputs_test.argmax(dim=-1) == label_test).float().sum().item()

                # Store predictions and labels
                all_labels.extend(label_test.cpu().numpy())
                all_predictions.extend(outputs_test.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / len(test_loader.dataset)

        print(f'Test Results: Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
        return {
            'test_loss': avg_test_loss,
            'test_accuracy': test_accuracy,
            'all_labels': all_labels,
            'all_predictions': all_predictions
        }
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves as in original notebook."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].plot(self.loss_list, label="Train Loss", color="blue", linewidth=2)
        axes[0].plot(self.val_loss_list, label="Validation Loss", color="red", linewidth=2)
        axes[0].set_title("Loss over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(self.accuracy_list, label="Train Accuracy", color="blue", linewidth=2)
        axes[1].plot(self.val_accuracy_list, label="Validation Accuracy", color="red", linewidth=2)
        axes[1].set_title("Accuracy over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def save_results(self, test_results: Dict[str, Any], output_dir: str = None):
        """Save test results for evaluation (as in original notebook)."""
        import pickle
        
        if output_dir is None:
            output_dir = self.config['output']['root_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as in original notebook
        with open(os.path.join(output_dir, "all_labels.pkl"), "wb") as f:
            pickle.dump(test_results['all_labels'], f)

        with open(os.path.join(output_dir, "all_predictions.pkl"), "wb") as f:
            pickle.dump(test_results['all_predictions'], f)
            
        print(f"Results saved to {output_dir}")


def train_eeg_classifier(config_path: str = "config/config.yaml") -> ClassifierTrainer:
    """
    Complete training pipeline for EEG classification.
    Replicates the exact workflow from original eeg_classification.py notebook.
    """
    from data import load_config
    
    # Set seed early like original notebook (before any operations)
    seed = 45  # Same as original notebook
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Always call like original
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_loader = CATVisDataLoader(config)
    df = data_loader.get_dataset_dataframe()
    train_df, val_df, test_df = data_loader.get_train_val_test_splits(df)
    
    print(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets and data loaders
    preprocessor = DataPreprocessor(config)
    train_dataset, val_dataset, test_dataset = preprocessor.create_classification_datasets(
        train_df, val_df, test_df
    )
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Initialize trainer
    trainer = ClassifierTrainer(config, device)
    
    # Train model
    training_history = trainer.train(train_loader, val_loader)
    
    # Test model
    test_results = trainer.test(test_loader)
    
    # Plot and save results
    trainer.plot_training_curves()
    trainer.save_results(test_results)
    
    return trainer 