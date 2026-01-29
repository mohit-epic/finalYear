#!/usr/bin/env python3
"""
Check raw SEN12MS-CR data values before normalization
"""

import tifffile
import numpy as np

import glob
import os
import sys

# Find any TIF file in the input directory
search_path = "/kaggle/input/sen12ms-cr-winter/**/*.tif"
tif_files = glob.glob(search_path, recursive=True)

if not tif_files:
    print(f"No TIF files found in {search_path}")
    # Try alternate path just in case
    search_path_alt = "/kaggle/input/**/*.tif"
    tif_files = glob.glob(search_path_alt, recursive=True)
    
if not tif_files:
    print("Could not find any TIF files to analyze!")
    sys.exit(1)

img_path = tif_files[0]
print(f"Analyzing file: {img_path}")

img = tifffile.imread(img_path)
print(f"\nImage shape: {img.shape}")

# Handle different formats
if img.ndim == 3:
    h, w, c = img.shape
    if c <= 20 and h > c and w > c:
        img = np.transpose(img, (2, 0, 1))

# Extract RGB bands [B4, B3, B2] = indices [3, 2, 1]
rgb = img[[3, 2, 1], :, :]

print(f"\nRaw RGB values:")
print(f"  Min: {rgb.min()}")
print(f"  Max: {rgb.max()}")
print(f"  Mean: {rgb.mean():.2f}")
print(f"  Median: {np.median(rgb):.2f}")
print(f"  Percentiles:")
print(f"    1%: {np.percentile(rgb, 1):.2f}")
print(f"    5%: {np.percentile(rgb, 5):.2f}")
print(f"    95%: {np.percentile(rgb, 95):.2f}")
print(f"    99%: {np.percentile(rgb, 99):.2f}")

# Test different normalization approaches
print(f"\n--- Normalization Tests ---")

# Current approach: divide by 10000
norm1 = np.clip(rgb, 0, 10000) / 10000.0
print(f"\nDivide by 10000:")
print(f"  Range: [{norm1.min():.4f}, {norm1.max():.4f}]")
print(f"  Mean: {norm1.mean():.4f}")

# Alternative: percentile-based
p2, p98 = np.percentile(rgb, [2, 98])
norm2 = np.clip((rgb - p2) / (p98 - p2), 0, 1)
print(f"\nPercentile stretch (2-98%):")
print(f"  Range: [{norm2.min():.4f}, {norm2.max():.4f}]")
print(f"  Mean: {norm2.mean():.4f}")

# Alternative: divide by max
norm3 = rgb / rgb.max()
print(f"\nDivide by max ({rgb.max()}):")
print(f"  Range: [{norm3.min():.4f}, {norm3.max():.4f}]")
print(f"  Mean: {norm3.mean():.4f}")

print("\n" + "=" * 60)
