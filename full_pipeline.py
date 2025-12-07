#!/usr/bin/env python3

"""
Complete Heritage Semantic-NeRF Pipeline
Entirely original: Custom SfM, Custom Segmenter, Custom NeRF
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command and handle errors"""
    print(f"\n{'='*70}")
    print(f"[*] {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False

def run_full_pipeline():
    """Execute complete original pipeline"""
    
    print("\n" + "="*70)
    print("HERITAGE SEMANTIC-NERF: COMPLETE ORIGINAL IMPLEMENTATION")
    print("="*70)
    print("\nAll components are 100% originally developed:")
    print("  • Custom Structure-from-Motion (SfM)")
    print("  • Custom Semantic Segmentation Network")
    print("  • Custom Semantic-NeRF Architecture")
    print("  • Custom Volume Rendering with Semantic Consistency")
    print("\n" + "="*70)
    
    # Step 1: Train Feature Extractor (optional, for custom SfM)
    print("\n[OPTIONAL] Step 0: Train feature extractor for SfM")
    print("(Skip if using pre-trained features)")
    if input("Train feature extractor? (y/n): ").lower() == 'y':
        run_command(
            "python sfm/train_sfm.py",
            "Training SfM feature extractor"
        )
    
    # Step 2: Run Custom SfM
    print("\n[STEP 1/4] Running custom Structure-from-Motion...")
    if not run_command(
        "python sfm/sfm_pipeline.py",
        "Custom SfM reconstruction"
    ):
        print("ERROR: SfM failed. Check your input data.")
        return False
    
    # Step 3: Train Semantic Segmenter
    print("\n[STEP 2/4] Training custom semantic segmentation network...")
    if not run_command(
        "python segmentation/train_segmenter.py",
        "Training semantic segmenter"
    ):
        print("ERROR: Segmenter training failed.")
        return False
    
    # Step 4: Generate Semantic Masks
    print("\n[STEP 3/4] Generating semantic masks...")
    if not run_command(
        "python segmentation/inference.py",
        "Generating semantic masks"
    ):
        print("ERROR: Mask generation failed.")
        return False
    
    # Step 5: Train Semantic-NeRF
    print("\n[STEP 4/4] Training custom Semantic-NeRF...")
    if not run_command(
        "python train_nerf.py",
        "Training Semantic-NeRF"
    ):
        print("ERROR: NeRF training failed.")
        return False
    
    # Success
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  • checkpoints/: Trained model weights")
    print("  • outputs/: Rendered views, semantic maps, logs")
    print("\nKey achievements:")
    print("  ✓ Custom SfM extracts camera poses")
    print("  ✓ Custom segmenter identifies heritage elements")
    print("  ✓ Custom NeRF produces semantic 3D model")
    print("  ✓ Complete end-to-end system 100% original")
    print("\n" + "="*70)
    
    return True

if __name__ == '__main__':
    success = run_full_pipeline()
    sys.exit(0 if success else 1)
