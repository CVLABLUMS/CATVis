#!/usr/bin/env python3
"""
Run CATVis Image Generation Pipeline.
Usage: python scripts/run_pipeline.py [--config CONFIG_PATH] [--subjects SUBJECT_IDS] [--max-batches N]
"""

import sys
import os
import argparse
import torch
import numpy as np
from typing import List, Optional

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import load_config, CATVisDataLoader, DataPreprocessor, setup_deterministic_environment
from models import EEGClassifier, ContrastiveEncoder
from pipeline import TextRetrieval, ImageGenerator
from evaluation.metrics import evaluate_retrieval_performance


def run_catvis_pipeline(config_path: str = "config/config.yaml", 
                       subject_filter: Optional[List[int]] = None,
                       max_batches: Optional[int] = None) -> dict:
    """
    Run the complete CATVis pipeline.
    Replicates the workflow from original catvis_pipeline.py notebook.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup comprehensive deterministic environment
    setup_deterministic_environment(seed=config['seed'], use_deterministic_algorithms=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Subject filter: {subject_filter}")
    print(f"Max batches: {max_batches}")
    
    # Load data
    print("\n=== Loading Data ===")
    data_loader = CATVisDataLoader(config)
    df = data_loader.get_dataset_dataframe()
    train_df, val_df, test_df = data_loader.get_train_val_test_splits(df)
    
    print(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets for pipeline (use test split)
    preprocessor = DataPreprocessor(config)
    test_dataset = preprocessor.create_pipeline_dataset(test_df)
    test_loader = preprocessor.create_data_loaders(
        test_dataset, test_dataset, test_dataset,  # Use test for all
        batch_size=config['generation']['batch_size'],
        shuffle_train=False
    )[2]  # Get test loader
    
    # Load EEG Classifier
    print("\n=== Loading EEG Classifier ===")
    eeg_classifier = EEGClassifier(config).to(device)
    classifier_checkpoint = os.path.join(
        config['checkpoints']['root_dir'], 
        config['checkpoints']['eeg_classifier']
    )
    
    if not os.path.exists(classifier_checkpoint):
        raise FileNotFoundError(f"EEG classifier checkpoint not found: {classifier_checkpoint}")
    
    eeg_classifier.load_pretrained_weights(classifier_checkpoint)
    print(f"Loaded EEG classifier from: {classifier_checkpoint}")
    
    # Load Contrastive Model
    print("\n=== Loading Contrastive Model ===")
    contrastive_model = ContrastiveEncoder(config).to(device)
    contrastive_checkpoint = os.path.join(
        config['checkpoints']['root_dir'], 
        config['checkpoints']['contrastive_model']
    )
    
    if not os.path.exists(contrastive_checkpoint):
        raise FileNotFoundError(f"Contrastive model checkpoint not found: {contrastive_checkpoint}")
    
    contrastive_model.load_pretrained_weights(contrastive_checkpoint)
    print(f"Loaded contrastive model from: {contrastive_checkpoint}")
    
    # Setup Text Retrieval
    print("\n=== Setting Up Text Retrieval ===")
    text_retrieval = TextRetrieval(config, contrastive_model, device)
    # Always use original full test set for retrieval corpus and evaluation (consistency)
    text_retrieval.setup_retrieval_corpus(test_df)
    
    # Evaluate retrieval performance on original full test set
    retrieval_results = evaluate_retrieval_performance(
        contrastive_model, test_df, device, config['contrastive_training']['clip_model']
    )
    
    # Setup Image Generator
    print("\n=== Setting Up Image Generator ===")
    image_generator = ImageGenerator(config, eeg_classifier, text_retrieval, device)
    
    # Generate Images
    print("\n=== Starting Image Generation ===")
    generation_results = image_generator.generate_for_test_set(
        test_loader, data_loader, subject_filter, max_batches
    )
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"Generated {len(generation_results['generated_images'])} images")
    print(f"Results saved to: {image_generator.output_dir}")
    
    return {
        'generation_results': generation_results,
        'retrieval_results': retrieval_results,
        'output_dir': image_generator.output_dir
    }


def main():
    parser = argparse.ArgumentParser(description='Run CATVis Image Generation Pipeline')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--subjects',
        type=str,
        help='Comma-separated list of subject IDs to process (e.g., "1,2,4"). If not specified, all subjects will be processed.'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        help='Maximum number of batches to process (for testing). If not specified, all batches will be processed.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run pipeline setup without generating images (for testing)'
    )
    
    args = parser.parse_args()
    
    # Parse subject filter
    subject_filter = None
    if args.subjects:
        try:
            subject_filter = [int(s.strip()) for s in args.subjects.split(',')]
        except ValueError:
            print("❌ Error: Invalid subject IDs. Please provide comma-separated integers.")
            return 1
    
    print("=== CATVis Image Generation Pipeline ===")
    print(f"Config: {args.config}")
    
    if args.dry_run:
        print("🔍 Dry run mode - setup only, no image generation")
        args.max_batches = 1  # Process only 1 batch for testing
    
    try:
        results = run_catvis_pipeline(
            config_path=args.config,
            subject_filter=subject_filter,
            max_batches=args.max_batches
        )
        
        if not args.dry_run:
            print(f"\n📊 Summary:")
            print(f"  Generated images: {len(results['generation_results']['generated_images'])}")
            print(f"  Output directory: {results['output_dir']}")
            print(f"  Text retrieval results: {results['retrieval_results']}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 