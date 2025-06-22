#!/usr/bin/env python3
"""
Evaluate CATVis Results.
Usage: python scripts/evaluate_results.py [--config CONFIG_PATH] [--results-dir RESULTS_DIR]
"""

import sys
import os
import argparse
import pickle
import torch
import pandas as pd
from typing import Optional

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import load_config, CATVisDataLoader, DataPreprocessor
from evaluation import EvaluationMetrics
from models import ContrastiveEncoder


def evaluate_contrastive_retrieval(config_path: str = "config/config.yaml") -> dict:
    """
    Evaluate contrastive model retrieval performance.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load contrastive model
    print("Loading contrastive model...")
    contrastive_model = ContrastiveEncoder(config).to(device)
    contrastive_checkpoint = os.path.join(
        config['checkpoints']['root_dir'], 
        config['checkpoints']['contrastive_model']
    )
    
    if not os.path.exists(contrastive_checkpoint):
        raise FileNotFoundError(f"Contrastive model checkpoint not found: {contrastive_checkpoint}")
    
    contrastive_model.load_pretrained_weights(contrastive_checkpoint)
    print(f"Loaded contrastive model from: {contrastive_checkpoint}")
    
    # Load data
    print("Loading data...")
    data_loader = CATVisDataLoader(config)
    df = data_loader.get_dataset_dataframe()
    train_df, val_df, test_df = data_loader.get_train_val_test_splits(df)
    
    # Create contrastive datasets
    preprocessor = DataPreprocessor(config)
    _, _, test_dataset = preprocessor.create_contrastive_datasets(
        train_df, val_df, test_df
    )
    
    # Create data loader for contrastive evaluation
    _, _, test_loader = preprocessor.create_data_loaders(
        test_dataset, test_dataset, test_dataset,
        batch_size=config['contrastive_training']['batch_size']
    )
    
    # Run retrieval evaluation using shared function
    from evaluation.metrics import evaluate_retrieval_performance
    retrieval_results = evaluate_retrieval_performance(
        contrastive_model, test_df, device, config['contrastive_training']['clip_model']
    )
    
    return retrieval_results


def evaluate_catvis_results(config_path: str = "config/config.yaml",
                           results_dir: str = None) -> dict:
    """
    Run comprehensive evaluation on CATVis results.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine results directory
    if results_dir is None:
        results_dir = config['output']['root_dir']
    
    print(f"Evaluating results in: {results_dir}")
    
    # Load data for mappings
    data_loader = CATVisDataLoader(config)
    df = data_loader.get_dataset_dataframe()
    train_df, val_df, test_df = data_loader.get_train_val_test_splits(df)
    
    # Try to load classification results if available
    all_labels = None
    all_predictions = None
    
    labels_file = os.path.join(results_dir, "all_labels.pkl")
    predictions_file = os.path.join(results_dir, "all_predictions.pkl")
    
    if os.path.exists(labels_file) and os.path.exists(predictions_file):
        print("Loading classification results...")
        with open(labels_file, "rb") as f:
            all_labels = pickle.load(f)
        with open(predictions_file, "rb") as f:
            all_predictions = pickle.load(f)
        print(f"Loaded {len(all_labels)} classification results")
    else:
        print("Classification results not found - skipping classification metrics")
    
    # Initialize evaluation metrics
    evaluator = EvaluationMetrics(config, device)
    
    # Run comprehensive evaluation
    eval_results = evaluator.run_comprehensive_evaluation(
        results_dir=results_dir,
        all_labels=all_labels,
        all_predictions=all_predictions,
        label_to_simple_class=data_loader.label_to_simple_class,
        test_df=test_df
    )
    
    # Create visual grid if requested
    generated_dir = os.path.join(results_dir, config['output']['generated_images'])
    gt_dir = os.path.join(results_dir, config['output']['ground_truth_images'])
    
    if os.path.exists(generated_dir) and os.path.exists(gt_dir):
        print("\n=== Creating Visual Grid ===")
        
        # Sample images for visualization (from original evaluation)
        selected_images = [
            '0901_cat.jpg', '1336_canoe.jpg', '0838_computer.jpg',
            '0604_parachute.jpg', '0193_pool.jpg', '0040_piano.jpg',
            '0102_parachute.jpg', '0749_guitar.jpg', '0964_bike.jpg',
            '0467_broom.jpg', '0435_golf.jpg', '0032_locomotive.jpg',
            '0801_mug.jpg', '0858_dog.jpg', '0053_chair.jpg',
            '0745_convertible.jpg', '0219_chair.jpg', '0981_cat.jpg'
        ]
        
        # Filter to only include images that exist
        existing_images = []
        for img in selected_images:
            if os.path.exists(os.path.join(gt_dir, img)):
                existing_images.append(img)
        
        if existing_images:
            visual_grid_path = os.path.join(results_dir, "visual_grid.png")
            grid_image = evaluator.create_visual_grid(
                existing_images[:12],  # Limit to 12 images
                generated_dir, gt_dir,
                subject=4,  # Default subject
                grid_cols=3,
                save_path=visual_grid_path
            )
            print(f"Visual grid saved to: {visual_grid_path}")
    
    return eval_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CATVis Results')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        help='Directory containing results to evaluate. If not specified, uses output.root_dir from config.'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        choices=['all', 'classification', 'generation'],
        default='all',
        help='Which metrics to compute (default: all)'
    )
    parser.add_argument(
        '--contrastive',
        action='store_true',
        help='Evaluate contrastive model retrieval performance only'
    )
    
    args = parser.parse_args()
    
    print("=== CATVis Results Evaluation ===")
    print(f"Config: {args.config}")
    
    try:
        if args.contrastive:
            print("Mode: Contrastive retrieval evaluation only")
            eval_results = evaluate_contrastive_retrieval(config_path=args.config)
        else:
            print(f"Metrics: {args.metrics}")
            eval_results = evaluate_catvis_results(
                config_path=args.config,
                results_dir=args.results_dir
            )
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"\n📊 Results Summary:")
        
        if args.contrastive:
            # Handle contrastive results (recall metrics)
            for metric, value in eval_results.items():
                print(f"  {metric}: {value:.2f}%")
        else:
            # Handle general evaluation results
            for category, metrics in eval_results.items():
                print(f"\n{category.upper()}:")
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
                else:
                    print(f"  {metrics}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 