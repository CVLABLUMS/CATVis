"""
Image Generation Module for CATVis Pipeline.
Extracted from original catvis_pipeline.py notebook.
Implements Stable Diffusion with Beta Prior Semantic Interpolation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import shutil
from PIL import Image
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

from models import EEGClassifier
from pipeline.retrieval import TextRetrieval


class ImageGenerator:
    """
    Image generation system using Stable Diffusion with Beta Prior Semantic Interpolation.
    Extracted from original catvis_pipeline.py with exact methodology preserved.
    """
    
    def __init__(self, config: Dict[str, Any], eeg_classifier: EEGClassifier, 
                 text_retrieval: TextRetrieval, device: torch.device):
        self.config = config
        self.device = device
        self.eeg_classifier = eeg_classifier
        self.text_retrieval = text_retrieval
        
        # Generation parameters
        gen_config = config['generation']
        self.batch_size = gen_config['batch_size']
        self.num_samples = gen_config['num_samples']
        self.num_inference_steps = gen_config['num_inference_steps']
        self.guidance_scale = gen_config['guidance_scale']
        self.beta_alpha = gen_config['beta_alpha']
        self.beta_beta = gen_config['beta_beta']
        
        # Setup Stable Diffusion components
        self._setup_stable_diffusion(gen_config['stable_diffusion_model'])
        
        # Beta distribution for interpolation
        self.beta_dist = torch.distributions.Beta(self.beta_alpha, self.beta_beta)
        
        # Generator for reproducibility
        if torch.cuda.is_available():
            self.cuda_generator = torch.Generator(device=device)
            self.cuda_generator.manual_seed(config['seed'])
        else:
            self.cuda_generator = None
            
        # Output directories
        self.output_config = config['output']
        self.output_dir = self.output_config['root_dir']
        self.generated_dir = os.path.join(self.output_dir, self.output_config['generated_images'])
        self.gt_dir = os.path.join(self.output_dir, self.output_config['ground_truth_images'])
        
        # Create output directories
        os.makedirs(self.generated_dir, exist_ok=True)
        os.makedirs(self.gt_dir, exist_ok=True)
        
        # Track generated images and their retrieved captions
        self.generation_track = {}
        self.track_file = os.path.join(self.output_dir, "generation_track.pkl")
        
    def _setup_stable_diffusion(self, model_id: str):
        """Setup Stable Diffusion components as in original pipeline."""
        print("Loading Stable Diffusion components...")
        
        # Load components
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", variant="fp16", torch_dtype=torch.float16
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", variant="fp16", torch_dtype=torch.float16
        )
        self.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # Move to device
        self.vae = self.vae.to(self.device)
        self.unet = self.unet.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        
        print("Stable Diffusion components loaded successfully")
        
    def _custom_sd_pipe(self, class_embedding: torch.Tensor, caption_embedding: torch.Tensor) -> Image.Image:
        """
        Custom Stable Diffusion pipeline with beta prior interpolation.
        Extracted exactly from original catvis_pipeline.py.
        """
        # Interpolated Conditional Text Embeddings
        interp_coef = self.beta_dist.sample()
        mean_text_embeddings = interp_coef * class_embedding + (1 - interp_coef) * caption_embedding
        
        # Prepare unconditional embeddings
        max_length = mean_text_embeddings.shape[-1]  # Use the actual max length
        uncond_input = self.tokenizer(
            [""] * self.batch_size, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        cond_embeddings = torch.cat([uncond_embeddings, mean_text_embeddings])
        cond_embeddings = cond_embeddings.half()  # fp16 half precision

        # Latent noise
        latents = torch.randn(
            (self.batch_size, self.unet.config.in_channels, 64, 64),
            generator=self.cuda_generator,
            device=self.device,
            dtype=torch.float16
        )
        latents = latents * self.scheduler.init_noise_sigma

        # Denoising loop
        self.scheduler.set_timesteps(self.num_inference_steps)
        for t in self.scheduler.timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            
            # Predict noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=cond_embeddings).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode with VAE
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            img = self.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (img * 255).round().astype("uint8")
        pil_images = [Image.fromarray(img) for img in images]

        return pil_images[0]
    
    def generate_images_for_batch(self, test_batch: Dict[str, Any], 
                                data_loader, subject_filter: List[int] = None) -> Dict[str, Any]:
        """
        Generate images for a batch of EEG data.
        Replicates the generation logic from original pipeline.
        """
        results = {
            'generated_images': [],
            'ground_truth_paths': [],
            'predicted_classes': [],
            'true_classes': [],
            'retrieved_captions': [],
            'generation_metadata': []
        }
        
        # Extract batch data
        eeg_batch = test_batch['eeg'].to(self.device)
        label_batch = test_batch['label']
        image_batch = test_batch['image']
        subject_batch = test_batch['subject']
        caption_batch = test_batch['caption']
        
        batch_size = eeg_batch.size(0)
        
        for i in range(batch_size):
            eeg_sample = eeg_batch[i:i+1]  # Keep batch dimension
            true_label = label_batch[i].item()
            image_idx = image_batch[i].item()
            subject_id = subject_batch[i].item()
            true_caption = caption_batch[i]
            
            # Skip if subject filter is specified and this subject is not included
            if subject_filter is not None and subject_id not in subject_filter:
                continue
                
            # EEG Classification
            self.eeg_classifier.eval()
            with torch.no_grad():
                outputs, _ = self.eeg_classifier(eeg_sample)
                prediction = outputs.argmax(dim=-1).cpu().numpy()[0]
            
            # Get predicted class prompt
            predicted_prompt = data_loader.label_to_class[prediction]
            true_class_name = data_loader.label_to_simple_class[true_label]
            
            # Caption retrieval
            top_k_results = self.text_retrieval.retrieve_top_k_from_eeg(
                eeg_sample.squeeze(0), k=10
            )
            
            # Generate class and caption embeddings
            class_prompt = [predicted_prompt]
            text_input = self.tokenizer(
                class_prompt, 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            class_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]
            
            # Get caption embeddings for retrieved captions
            retrieved_captions = [result[0] for result in top_k_results]
            text_input2 = self.tokenizer(
                retrieved_captions, 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            caption_embeds = self.text_encoder(text_input2.input_ids.to(self.device))[0]
            
            # Re-ranking based on class-caption similarity
            query = class_embeds.view(1, -1)
            key = caption_embeds.view(len(retrieved_captions), -1)
            sim = F.cosine_similarity(query, key, dim=1)
            top_k_indices = torch.topk(sim, k=self.num_samples, largest=True).indices.tolist()
            
            # Copy ground truth image
            self._save_ground_truth_image(image_idx, true_class_name, data_loader)
            
            # Generate images
            generated_paths = []
            used_captions = []
            
            for sample_idx in range(self.num_samples):
                # Generate image name
                gt_name = f"{image_idx:04d}_{true_class_name}"
                predicted_class_simple = data_loader.label_to_simple_class[prediction]
                gene_image_name = f"{gt_name}_s{subject_id}_{predicted_class_simple}_{sample_idx}.png"
                
                gene_image_path = os.path.join(self.generated_dir, gene_image_name)
                
                if os.path.exists(gene_image_path):
                    print(f"Skipping {gene_image_name} (already exists)")
                    generated_paths.append(gene_image_path)
                    used_captions.append(retrieved_captions[top_k_indices[sample_idx]])
                    continue
                
                # Generate image
                caption_idx = top_k_indices[sample_idx]
                selected_caption = retrieved_captions[caption_idx]
                
                generated_image = self._custom_sd_pipe(class_embeds, caption_embeds[caption_idx:caption_idx+1])
                generated_image.save(gene_image_path)
                
                # Track generation
                self.generation_track[gene_image_name] = {
                    'retrieved_caption': selected_caption,
                    'predicted_class': predicted_prompt,
                    'true_class': true_class_name,
                    'subject_id': subject_id,
                    'image_idx': image_idx,
                    'caption_score': top_k_results[caption_idx][1]
                }
                
                generated_paths.append(gene_image_path)
                used_captions.append(selected_caption)
                
                # Save tracking info incrementally
                self._save_tracking_info()
            
            # Store results
            results['generated_images'].extend(generated_paths)
            results['predicted_classes'].append(predicted_prompt)
            results['true_classes'].append(true_class_name)
            results['retrieved_captions'].extend(used_captions)
            results['generation_metadata'].append({
                'image_idx': image_idx,
                'subject_id': subject_id,
                'true_label': true_label,
                'predicted_label': prediction,
                'all_retrieved_captions': [r[0] for r in top_k_results]
            })
            
        return results
    
    def _save_ground_truth_image(self, image_idx: int, class_name: str, data_loader):
        """Save ground truth image to GT directory."""
        gt_path = data_loader.image_to_path[image_idx]
        gt_name = f"{image_idx:04d}_{class_name}"
        
        # Source and destination paths
        source = os.path.join(
            data_loader.config['data']['root_dir'],
            data_loader.config['data']['imagenet_images'],
            gt_path
        )
        destination = os.path.join(self.gt_dir, f"{gt_name}.jpg")
        
        if not os.path.exists(destination) and os.path.exists(source):
            shutil.copyfile(source, destination)
    
    def _save_tracking_info(self):
        """Save generation tracking information incrementally."""
        with open(self.track_file + '.tmp', 'wb') as temp_file:
            pickle.dump(self.generation_track, temp_file)
        os.rename(self.track_file + '.tmp', self.track_file)
    
    def generate_for_test_set(self, test_loader, data_loader, 
                            subject_filter: List[int] = None, max_batches: int = None) -> Dict[str, Any]:
        """
        Generate images for entire test set.
        
        Args:
            test_loader: DataLoader for test data
            data_loader: CATVisDataLoader instance for mappings
            subject_filter: List of subject IDs to include (None for all)
            max_batches: Maximum number of batches to process (None for all)
        """
        print("Starting image generation for test set...")
        
        all_results = {
            'generated_images': [],
            'ground_truth_paths': [],
            'predicted_classes': [],
            'true_classes': [],
            'retrieved_captions': [],
            'generation_metadata': []
        }
        
        processed_batches = 0
        
        for batch in tqdm(test_loader, desc="Generating images"):
            # Generate for this batch
            batch_results = self.generate_images_for_batch(batch, data_loader, subject_filter)
            
            # Aggregate results
            for key in all_results:
                all_results[key].extend(batch_results[key])
            
            processed_batches += 1
            if max_batches is not None and processed_batches >= max_batches:
                break
        
        print(f"Generated {len(all_results['generated_images'])} images")
        print(f"Results saved to {self.output_dir}")
        
        return all_results
    
    def cleanup_outputs(self):
        """Clean up output directories (for testing purposes)."""
        if os.path.exists(self.output_dir):
            print(f"Cleaning up output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir) 