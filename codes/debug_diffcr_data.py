#!/usr/bin/env python3
"""
Debug DiffCR output to understand why metrics are so low
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/kaggle/working/finalYear/codes/DiffCR')

from data.sen12ms_adapter import SEN12MS_CR_RGB

# Load one sample
dataset = SEN12MS_CR_RGB(
    data_root="/kaggle/input/sen12ms-cr-winter",
    mode="test",
    data_csv="/kaggle/working/data.csv"
)

print("=" * 60)
print("DiffCR Data Debug")
print("=" * 60)

# Get first sample
sample = dataset[0]

gt_image = sample['gt_image']
cond_image = sample['cond_image']

print(f"\nGround Truth Image:")
print(f"  Shape: {gt_image.shape}")
print(f"  Min: {gt_image.min():.4f}")
print(f"  Max: {gt_image.max():.4f}")
print(f"  Mean: {gt_image.mean():.4f}")
print(f"  Std: {gt_image.std():.4f}")

print(f"\nConditional Image (stacked):")
print(f"  Shape: {cond_image.shape}")
print(f"  Min: {cond_image.min():.4f}")
print(f"  Max: {cond_image.max():.4f}")
print(f"  Mean: {cond_image.mean():.4f}")
print(f"  Std: {cond_image.std():.4f}")

# Check if values are in expected range [-1, 1]
if gt_image.min() < -1.5 or gt_image.max() > 1.5:
    print("\n⚠️  WARNING: Ground truth values outside [-1, 1] range!")
    
if cond_image.min() < -1.5 or cond_image.max() > 1.5:
    print("\n⚠️  WARNING: Conditional image values outside [-1, 1] range!")

# Test normalization conversion
gt_01 = ((gt_image.numpy() + 1) / 2.0).clip(0, 1)
print(f"\nGround Truth after [0,1] conversion:")
print(f"  Min: {gt_01.min():.4f}")
print(f"  Max: {gt_01.max():.4f}")
print(f"  Mean: {gt_01.mean():.4f}")

print("\n" + "=" * 60)
