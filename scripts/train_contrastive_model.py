#!/usr/bin/env python3
"""
Train Contrastive EEG-Text Model for CATVis.
Usage: python scripts/train_contrastive_model.py [--config CONFIG_PATH]
"""

import sys
import os
import argparse

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training import train_contrastive_model


def main():
    parser = argparse.ArgumentParser(description='Train Contrastive EEG-Text Model for CATVis')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    print("=== CATVis Contrastive Training ===")
    print(f"Using config: {args.config}")
    
    try:
        trainer = train_contrastive_model(args.config)
        print("\n✅ Contrastive training completed successfully!")
        print(f"Best model saved to: {trainer.model_save_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 