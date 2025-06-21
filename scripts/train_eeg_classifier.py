#!/usr/bin/env python3
"""
Train EEG Classifier for CATVis.
Usage: python scripts/train_eeg_classifier.py [--config CONFIG_PATH]
"""

import sys
import os
import argparse

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training import train_eeg_classifier


def main():
    parser = argparse.ArgumentParser(description='Train EEG Classifier for CATVis')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    print("=== CATVis EEG Classification Training ===")
    print(f"Using config: {args.config}")
    
    try:
        trainer = train_eeg_classifier(args.config)
        print("\n✅ EEG Classification training completed successfully!")
        print(f"Best model saved to: {trainer.model_save_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 