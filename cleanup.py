#!/usr/bin/env python3
"""
Cleanup script for CATVis generated files.
Simple approach: Remove entire outputs directory, selectively remove checkpoints.
"""

import os
import argparse
import shutil
from pathlib import Path


def remove_file_if_exists(filepath: str, verbose: bool = True) -> bool:
    """Remove a file if it exists."""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            if verbose:
                print(f"✅ Removed: {filepath}")
            return True
        except Exception as e:
            if verbose:
                print(f"❌ Failed to remove {filepath}: {e}")
            return False
    else:
        if verbose:
            print(f"⏭️  Not found: {filepath}")
        return False


def remove_directory_if_exists(directory: str, verbose: bool = True) -> bool:
    """Remove a directory and all its contents if it exists."""
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            if verbose:
                print(f"✅ Removed directory: {directory}")
            return True
        except Exception as e:
            if verbose:
                print(f"❌ Failed to remove {directory}: {e}")
            return False
    else:
        if verbose:
            print(f"⏭️  Directory not found: {directory}")
        return False


def cleanup_outputs(verbose: bool = True) -> bool:
    """Clean up the entire outputs directory."""
    if verbose:
        print("🧹 Cleaning outputs directory...")
    
    return remove_directory_if_exists("outputs", verbose)


def cleanup_checkpoints(verbose: bool = True) -> int:
    """Clean up specific model checkpoints."""
    if verbose:
        print("🧹 Cleaning specific model checkpoints...")
    
    count = 0
    checkpoints_dir = "checkpoints"
    
    # Only remove the specific CATVis checkpoints
    checkpoint_files = [
        "eeg_classifier_best.pth",
        "contrastive_model_best.pth"
    ]
    
    for filename in checkpoint_files:
        if remove_file_if_exists(os.path.join(checkpoints_dir, filename), verbose):
            count += 1
    
    return count


def main():
    parser = argparse.ArgumentParser(description='Clean up CATVis generated files')
    parser.add_argument(
        '--mode',
        choices=['outputs', 'checkpoints', 'all'],
        default='outputs',
        help='What to clean up (default: outputs only)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true', 
        help='Show what would be removed without actually removing'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output messages'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if args.dry_run:
        print("🔍 DRY RUN - No files will be actually removed\n")
        verbose = True
    
    if not args.dry_run:
        # Confirm for destructive operations
        if args.mode == 'all':
            print("⚠️  This will remove the outputs directory and CATVis checkpoints.")
            confirm = input("Are you sure? (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Cleanup cancelled.")
                return
        elif args.mode == 'checkpoints':
            print("⚠️  This will remove CATVis model checkpoints.")
            confirm = input("Are you sure? (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Cleanup cancelled.")
                return
    
    removed_items = 0
    
    if args.mode in ['outputs', 'all']:
        if not args.dry_run:
            if cleanup_outputs(verbose):
                removed_items += 1
        else:
            if verbose:
                print("🔍 Would remove: outputs/ directory")
                removed_items += 1
    
    if args.mode in ['checkpoints', 'all']:
        if not args.dry_run:
            removed_items += cleanup_checkpoints(verbose)
        else:
            if verbose:
                print("🔍 Would remove: checkpoints/eeg_classifier_best.pth")
                print("🔍 Would remove: checkpoints/contrastive_model_best.pth")
                removed_items += 2
    
    if not args.dry_run:
        if verbose:
            print(f"\n✨ Cleanup complete! Removed {removed_items} items.")
    else:
        if verbose:
            print(f"\n🔍 Dry run complete! Would remove {removed_items} items.")


if __name__ == "__main__":
    main() 