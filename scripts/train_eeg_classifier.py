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
    parser = argparse.ArgumentParser(description='Train or Test EEG Classifier for CATVis')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Test existing checkpoint without training (requires existing model checkpoint)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file for testing (if not provided, uses config checkpoint path)'
    )
    
    args = parser.parse_args()
    
    if args.test_only:
        print("=== CATVis EEG Classification Testing ===")
        print(f"Using config: {args.config}")
        if args.checkpoint:
            print(f"Testing checkpoint: {args.checkpoint}")
        else:
            print("Testing checkpoint from config")
    else:
        print("=== CATVis EEG Classification Training ===")
        print(f"Using config: {args.config}")
    
    try:
        trainer = train_eeg_classifier(
            config_path=args.config,
            test_only=args.test_only,
            checkpoint_path=args.checkpoint
        )
        
        if args.test_only:
            print("\n✅ EEG Classification testing completed successfully!")
        else:
            print("\n✅ EEG Classification training completed successfully!")
            print(f"Best model saved to: {trainer.model_save_path}")
        
        print(f"Results saved to outputs directory")
        
    except Exception as e:
        mode = "Testing" if args.test_only else "Training"
        print(f"\n❌ {mode} failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 