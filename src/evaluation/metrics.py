"""
Evaluation Metrics Module for CATVis.
Extracted from original catvis_pipeline_evaluations.py notebook.
"""

import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
from torchvision.models import ViT_H_14_Weights, vit_h_14
from torchmetrics.functional import accuracy
from torchvision.models.inception import inception_v3
from pytorch_fid import fid_score
from scipy.stats import entropy
import torchvision.transforms as transforms
import clip

from data import EEGTextDataset


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for CATVis.
    Extracted from original research code with exact methodologies preserved.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.eval_config = config['evaluation']
        
    def compute_classification_metrics(self, all_labels: List[int], 
                                     all_predictions: List[List[float]], 
                                     label_to_simple_class: Dict[int, str], 
                                     top_k: int = 5) -> Dict[str, float]:
        """
        Compute classification metrics: top-k accuracy and F1 score.
        Extracted from original evaluation code.
        """
        print("Computing classification metrics...")
        
        # Convert to tensors
        all_labels = torch.tensor(all_labels)
        all_predictions = torch.tensor(all_predictions)  # Shape: (N, num_classes)

        # Compute top-k predictions
        top_k_preds = all_predictions.topk(k=top_k, dim=1).indices  # Shape: (N, top_k)

        # Compute top-k accuracy
        metrics = {}
        for k in range(1, top_k + 1):
            correct = (top_k_preds[:, :k] == all_labels.unsqueeze(1)).any(dim=1)
            metrics[f"top-{k}_accuracy"] = correct.float().mean().item() * 100

        # Compute F1 score (use top-1 predictions for F1)
        top_1_preds = top_k_preds[:, 0]  # Shape: (N,)
        f1 = f1_score(all_labels.cpu(), top_1_preds.cpu(), average="weighted")
        metrics["f1_score"] = f1 * 100

        # Print results
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}%")
            
        return metrics
    
    def plot_confusion_matrix(self, all_labels: List[int], all_predictions: List[List[float]], 
                            label_to_simple_class: Dict[int, str], save_path: str = None):
        """Plot confusion matrix as in original evaluation."""
        print("Generating confusion matrix...")
        
        true_labels = np.array(all_labels)
        predicted_labels = np.argmax(all_predictions, axis=1)
        classes = [label_to_simple_class[label] for label in range(40)]

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=False,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            xticklabels=classes,
            yticklabels=classes,
            square=True,
            linewidths=0.5
        )

        ax.set_xlabel("Predicted Labels", fontsize=14)
        ax.set_ylabel("True Labels", fontsize=14)
        ax.set_title("Confusion Matrix", fontsize=16, pad=20)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
    def compute_n_way_top_k_accuracy(self, generated_dir: str, gt_dir: str, 
                                   test_df: pd.DataFrame = None) -> float:
        """
        Compute N-way Top-K Classification Accuracy of Generation (GA).
        Extracted from original evaluation with exact methodology preserved.
        """
        print("Computing N-way Top-K accuracy...")
        
        # Load ViT model for classification
        weights = ViT_H_14_Weights.DEFAULT
        vit_clf_model = vit_h_14(weights=weights)
        preprocess = weights.transforms()
        vit_clf_model = vit_clf_model.to(self.device)
        vit_clf_model.eval()
        
        # Parameters from config
        n_way = self.eval_config['n_way']
        num_trials = self.eval_config['num_trials']
        top_k = self.eval_config['top_k']
        
        def n_way_top_k_acc(pred, class_id, n_way, num_trials, top_k):
            """Helper function from original code."""
            pick_range = [i for i in np.arange(len(pred)) if i != class_id]
            acc_list = []
            for t in range(num_trials):
                idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
                pred_picked = torch.cat([pred[class_id].unsqueeze(0), pred[idxs_picked]])
                acc = accuracy(
                    pred_picked.unsqueeze(0), 
                    torch.tensor([0], device=pred.device),
                    task="multiclass",
                    num_classes=n_way,
                    top_k=top_k
                )
                acc_list.append(acc.item())
            return np.mean(acc_list), np.std(acc_list)

        acc_list = []
        gt_images_name = os.listdir(gt_dir)
        gt_images_name.sort()
        
        # Get unique subjects from test_df if provided
        if test_df is not None:
            unique_subjects = test_df['subject'].unique()
        else:
            # Infer subjects from generated image names
            import glob
            generated_files = glob.glob(f"{generated_dir}/*.png")
            unique_subjects = set()
            for f in generated_files:
                # Extract subject from filename: 1995_shoe_s4_cat_0.png -> subject 4
                parts = os.path.basename(f).split('_')
                for part in parts:
                    if part.startswith('s') and part[1:].isdigit():
                        unique_subjects.add(int(part[1:]))
            unique_subjects = list(unique_subjects)

        for subject in unique_subjects:
            print(f"Processing subject: {subject}")
            sub_acc_list = []
            
            for gt_name in gt_images_name:
                # Load GT image
                gt_path = os.path.join(gt_dir, gt_name)
                if not os.path.exists(gt_path):
                    continue
                    
                real_image = Image.open(gt_path).convert('RGB')
                
                # Find generated images for this GT image and subject
                import glob
                gt_base = gt_name.split('.')[0]  # Remove extension
                generated_images = glob.glob(f"{generated_dir}/{gt_base}_s{subject}_*.png")
                
                if not generated_images:
                    continue
                
                # Get GT class prediction
                gt = preprocess(real_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    gt_class_id = vit_clf_model(gt).squeeze(0).softmax(0).argmax().item()

                # Evaluate each generated image
                for generated_image_path in generated_images:
                    if not os.path.exists(generated_image_path):
                        continue
                        
                    generated_image = Image.open(generated_image_path).convert('RGB')
                    pred = preprocess(generated_image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        pred_out = vit_clf_model(pred).squeeze(0).softmax(0).detach()
                    
                    acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
                    acc_list.append(acc)
                    sub_acc_list.append(acc)

            if sub_acc_list:
                print(f"Subject {subject} Accuracy: {np.mean(sub_acc_list)*100:.2f}%")
        
        average_accuracy = np.mean(acc_list) * 100 if acc_list else 0.0
        print(f"Average N-way Top-K Accuracy: {average_accuracy:.2f}%")
        
        return average_accuracy
    
    def compute_inception_score(self, generated_dir: str) -> float:
        """
        Compute Inception Score (IS).
        Extracted from original evaluation with exact methodology preserved.
        """
        print("Computing Inception Score...")
        
        batch_size = self.eval_config['inception_batch_size']
        
        def readDir(dirPath=generated_dir):
            """Helper function from original code."""
            allFiles = []
            if os.path.isdir(dirPath):
                fileList = os.listdir(dirPath)
                for f in fileList:
                    f = os.path.join(dirPath, f)
                    if os.path.isdir(f):
                        subFiles = readDir(f)
                        allFiles.extend(subFiles)
                    else:
                        if "_gt" not in f:  # Exclude ground truth files
                            allFiles.append(f)
                return allFiles
            else:
                return []

        def imread(filename):
            """Load image as numpy array."""
            return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]

        # Load inception model
        inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        inception_model.eval()
        up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(self.device)

        def get_pred(x):
            """Get predictions from inception model."""
            x = up(x)
            x = inception_model(x)
            return F.softmax(x, dim=1).data.cpu().numpy()

        # Get predictions
        files = readDir()
        if not files:
            print("No generated images found!")
            return 0.0
            
        N = len(files)
        preds = np.zeros((N, 1000))
        
        if batch_size > N:
            batch_size = N

        for i in tqdm(range(0, N, batch_size), desc="Computing predictions"):
            start = i
            end = min(i + batch_size, N)
            images = np.array([imread(str(f)).astype(np.float32) for f in files[start:end]])

            # Reshape to (n_images, 3, height, width)
            images = images.transpose((0, 3, 1, 2))
            images /= 255

            batch = torch.from_numpy(images).type(torch.FloatTensor).to(self.device)
            
            with torch.no_grad():
                y = get_pred(batch)
            preds[i:end] = y

        # Compute KL Divergence
        print('Computing KL Divergence...')
        py = np.mean(preds, axis=0)  # marginal probability
        scores = []
        for i in range(preds.shape[0]):
            pyx = preds[i, :]  # conditional probability
            scores.append(entropy(pyx, py))  # compute divergence

        mean_kl = np.mean(scores)
        inception_score = np.exp(mean_kl)
        
        print(f'Inception Score: {inception_score:.4f}')
        return inception_score
    
    def compute_fid_score(self, generated_dir: str, gt_dir: str) -> float:
        """
        Compute Fréchet Inception Distance (FID).
        Extracted from original evaluation with exact methodology preserved.
        """
        print("Computing FID score...")
        
        batch_size = self.eval_config['fid_batch_size']
        dims = self.eval_config['fid_dims']
        
        # Create temporary directory for resized GT images
        temp_path = os.path.join(os.path.dirname(gt_dir), "temp_gt_resized")
        os.makedirs(temp_path, exist_ok=True)
        
        # Resize GT images to 512x512 (as in original code)
        transform = transforms.Compose([transforms.Resize((512, 512))])
        
        try:
            for filename in os.listdir(gt_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(gt_dir, filename)
                    dest_path = os.path.join(temp_path, filename)
                    
                    with Image.open(src_path) as img:
                        transformed_img = transform(img)
                        transformed_img.save(dest_path)

            # Compute FID
            fid_value = fid_score.calculate_fid_given_paths(
                [generated_dir, temp_path], 
                batch_size=batch_size, 
                device=self.device, 
                dims=dims
            )
            
            print(f'FID Score: {fid_value:.4f}')
            
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
        
        return fid_value
    
    def create_visual_grid(self, selected_images: List[str], generated_dir: str, 
                          gt_dir: str, subject: int = 4, grid_cols: int = 3, 
                          image_size: Tuple[int, int] = (300, 300), 
                          save_path: str = None) -> Image.Image:
        """
        Create visual grid comparing GT and generated images.
        Extracted from original evaluation visualization code.
        """
        print("Creating visual grid...")
        
        from PIL import ImageDraw, ImageFont
        import glob
        
        spacing = 5
        group_spacing = 30
        images_per_group = 3  # GT + 2 samples
        num_groups = len(selected_images)

        # Calculate grid size
        grid_rows = (num_groups + grid_cols - 1) // grid_cols
        grid_width = grid_cols * images_per_group * image_size[0] + (grid_cols - 1) * spacing + (grid_cols - 1) * group_spacing
        grid_height = grid_rows * (image_size[1] + spacing) + (grid_rows - 1) * group_spacing + 50

        # Create canvas
        grid_image = Image.new("RGB", (grid_width, grid_height), "white")
        draw = ImageDraw.Draw(grid_image)

        # Load default font
        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Draw dashed vertical lines
        line_color = (64, 64, 64)
        line_width = 5
        dash_length = 20

        for col in range(1, grid_cols):
            x_pos = col * images_per_group * image_size[0] + col * spacing + col * group_spacing - group_spacing // 2 + 1
            for y in range(0, grid_height, dash_length * 2):
                draw.line((x_pos, y-70, x_pos, min(y-70 + dash_length, grid_height)), 
                         fill=line_color, width=line_width)

        # Place images
        for i, image_name in enumerate(selected_images):
            group_col = i % grid_cols
            group_row = i // grid_cols

            x_offset = group_col * images_per_group * image_size[0] + group_col * spacing + group_col * group_spacing
            y_offset = group_row * (image_size[1] + spacing) + group_row * group_spacing

            # Load images
            gt_path = os.path.join(gt_dir, image_name)
            base_name = image_name.split('.')[0]
            generated_paths = glob.glob(f"{generated_dir}/{base_name}_s{subject}_*_[01].png")
            generated_paths.sort()

            try:
                if os.path.exists(gt_path):
                    gt_image = Image.open(gt_path).resize(image_size).convert("RGB")
                    grid_image.paste(gt_image, (x_offset, y_offset))
                    
                    # Red bounding box for GT
                    draw.rectangle(
                        [x_offset, y_offset, x_offset + image_size[0], y_offset + image_size[1]],
                        outline="red", width=5
                    )

                # Generated images
                for j, gen_path in enumerate(generated_paths[:2]):  # Only first 2 samples
                    if os.path.exists(gen_path):
                        gen_image = Image.open(gen_path).resize(image_size).convert("RGB")
                        x_pos = x_offset + (j + 1) * (image_size[0] + spacing)
                        grid_image.paste(gen_image, (x_pos, y_offset))

            except Exception as e:
                print(f"Error loading images for {image_name}: {e}")
                continue

        # Add labels
        text_y_offset = grid_height - 30
        for col in range(grid_cols):
            x_offset = col * images_per_group * image_size[0] + col * spacing + col * group_spacing
            
            if font:
                draw.text((x_offset + image_size[0] // 2, text_y_offset), "GT", 
                         fill="black", font=font, anchor="mm")
                draw.text((x_offset + image_size[0] + spacing + image_size[0] // 2, text_y_offset), 
                         "Sample 1", fill="black", font=font, anchor="mm")
                draw.text((x_offset + 2 * (image_size[0] + spacing) + image_size[0] // 2, text_y_offset), 
                         "Sample 2", fill="black", font=font, anchor="mm")

        if save_path:
            grid_image.save(save_path, dpi=(300, 300))
            print(f"Visual grid saved to {save_path}")

        return grid_image
    
    def run_comprehensive_evaluation(self, results_dir: str, 
                                   all_labels: List[int] = None,
                                   all_predictions: List[List[float]] = None,
                                   label_to_simple_class: Dict[int, str] = None,
                                   test_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with all metrics.
        
        Args:
            results_dir: Directory containing generated images, GT images, and results
            all_labels: Classification labels (if available)
            all_predictions: Classification predictions (if available)
            label_to_simple_class: Label to class name mapping
            test_df: Test dataframe for additional info
        """
        print("Running comprehensive evaluation...")
        
        # Define paths
        generated_dir = os.path.join(results_dir, self.config['output']['generated_images'])
        gt_dir = os.path.join(results_dir, self.config['output']['ground_truth_images'])
        
        eval_results = {}
        
        # 1. Classification metrics (if available)
        if all_labels and all_predictions and label_to_simple_class:
            print("\n=== CLASSIFICATION METRICS ===")
            classification_metrics = self.compute_classification_metrics(
                all_labels, all_predictions, label_to_simple_class
            )
            eval_results['classification'] = classification_metrics
            
            # Confusion matrix
            confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
            self.plot_confusion_matrix(
                all_labels, all_predictions, label_to_simple_class, confusion_matrix_path
            )
        
        # 2. N-way Top-K accuracy
        if os.path.exists(generated_dir) and os.path.exists(gt_dir):
            print("\n=== N-WAY TOP-K ACCURACY ===")
            nway_acc = self.compute_n_way_top_k_accuracy(generated_dir, gt_dir, test_df)
            eval_results['nway_topk_accuracy'] = nway_acc
            
            # 3. Inception Score
            print("\n=== INCEPTION SCORE ===")
            is_score = self.compute_inception_score(generated_dir)
            eval_results['inception_score'] = is_score
            
            # 4. FID Score
            print("\n=== FID SCORE ===")
            fid_score_val = self.compute_fid_score(generated_dir, gt_dir)
            eval_results['fid_score'] = fid_score_val
        
        # Save evaluation results
        results_file = os.path.join(results_dir, "evaluation_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(eval_results, f)
        
        print(f"\nEvaluation complete! Results saved to {results_file}")
        return eval_results 


def evaluate_retrieval_performance(contrastive_model, test_df, device: torch.device, 
                                 clip_model_name: str = "ViT-L/14") -> Dict[str, float]:
    """
    Consolidated retrieval evaluation function.
    Eliminates code duplication between training and pipeline scripts.
    
    Args:
        contrastive_model: Trained contrastive EEG model
        test_df: Test dataframe with EEG data and captions
        device: Device to run evaluation on
        clip_model_name: CLIP model name for text encoding
        
    Returns:
        Dictionary with recall@k metrics
    """
    print("Evaluating retrieval performance...")
    
    # Load CLIP model (if not already loaded)
    if not hasattr(evaluate_retrieval_performance, '_clip_model_cache'):
        clip_model, _ = clip.load(clip_model_name, device=device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        evaluate_retrieval_performance._clip_model_cache = clip_model
    
    clip_model = evaluate_retrieval_performance._clip_model_cache
    
    # Extract unique captions for retrieval corpus
    retrieval_df = test_df.drop_duplicates(subset=["captions"])
    retrieval_dataset = EEGTextDataset(retrieval_df)
    retrieval_dataloader = DataLoader(retrieval_dataset, batch_size=128, shuffle=False)
    
    # Get unique caption embeddings
    unique_caption_embeds, unique_captions = _extract_text_embeddings(
        retrieval_dataloader, clip_model, device
    )
    
    # Create test dataset for EEG embeddings
    test_dataset = EEGTextDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Get all EEG embeddings and true captions
    all_eeg_embeds, all_true_captions = _extract_eeg_embeddings(
        test_loader, contrastive_model, device
    )
    
    # Compute similarities
    all_eeg_embeds = all_eeg_embeds.to(device)
    unique_caption_embeds = unique_caption_embeds.to(device)
    
    sim_e2t = all_eeg_embeds @ unique_caption_embeds.t()  # [N, M]
    
    # Create caption to index mapping
    caption_to_idx = {cap: idx for idx, cap in enumerate(unique_captions)}
    
    # Get true text indices
    true_text_indices = []
    for cap in all_true_captions:
        correct_idx = caption_to_idx[cap]
        true_text_indices.append(correct_idx)
    true_text_indices = torch.tensor(true_text_indices, device=device)
    
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


def _extract_text_embeddings(dataloader, clip_model, device) -> Tuple[torch.Tensor, List[str]]:
    """Extract text embeddings using CLIP."""
    all_text_embeds = []
    all_text_labels = []

    with torch.no_grad():
        for eeg_batch, text_batch in tqdm(dataloader, desc="Extracting text embeddings"):
            text_tokens = clip.tokenize(text_batch, truncate=True).to(device)
            text_emb = clip_model.encode_text(text_tokens).float()
            text_emb = F.normalize(text_emb, dim=-1)

            all_text_embeds.append(text_emb.cpu())
            all_text_labels.extend(text_batch)

    all_text_embeds = torch.cat(all_text_embeds, dim=0)  # [N, 768]
    return all_text_embeds, all_text_labels


def _extract_eeg_embeddings(dataloader, contrastive_model, device) -> Tuple[torch.Tensor, List[str]]:
    """Extract EEG embeddings."""
    all_eeg_embeds = []
    all_true_captions = []

    contrastive_model.eval()
    with torch.no_grad():
        for eeg_batch, caption_batch in tqdm(dataloader, desc="Extracting EEG embeddings"):
            eeg_batch = eeg_batch.to(device)
            eeg_embeds = contrastive_model(eeg_batch)  # [B, 768]
            eeg_embeds = F.normalize(eeg_embeds, dim=-1)
            all_eeg_embeds.append(eeg_embeds.cpu())
            all_true_captions.extend(caption_batch)

    all_eeg_embeds = torch.cat(all_eeg_embeds, dim=0)  # [N, 768]
    return all_eeg_embeds, all_true_captions 