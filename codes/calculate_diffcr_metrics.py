#!/usr/bin/env python3
"""
Calculate PSNR and SSIM metrics from DiffCR output images
Use this if metrics weren't displayed during testing
"""

import os
import sys
import glob
import numpy as np
import tifffile
from tqdm import tqdm

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    max_value = 1.0
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """Calculate SSIM (simplified version)"""
    import cv2
    
    # Convert to uint8
    img1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
    img2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim_map.mean())

# Find output directory
results_dir = "/kaggle/working/finalYear/codes/DiffCR/experiments/sen12ms_cr_diffcr_test/results/test"

if not os.path.exists(results_dir):
    print(f"Results directory not found: {results_dir}")
    sys.exit(1)

# Find GT and output images
gt_files = sorted(glob.glob(os.path.join(results_dir, "GT_*.tif")))
out_files = sorted(glob.glob(os.path.join(results_dir, "Out_*.tif")))

if len(gt_files) == 0 or len(out_files) == 0:
    print("No output images found!")
    sys.exit(1)

print(f"Found {len(gt_files)} image pairs")
print("Calculating metrics...")

psnr_values = []
ssim_values = []
mae_values = []

for gt_file, out_file in tqdm(zip(gt_files, out_files), total=len(gt_files)):
    # Load images
    gt_img = tifffile.imread(gt_file).astype(np.float32)
    out_img = tifffile.imread(out_file).astype(np.float32)
    
    # Normalize to [0, 1]
    gt_img = (gt_img + 1) / 2.0  # DiffCR outputs are in [-1, 1]
    out_img = (out_img + 1) / 2.0
    
    # Ensure shape (H, W, C)
    if gt_img.ndim == 3 and gt_img.shape[0] == 3:
        gt_img = np.transpose(gt_img, (1, 2, 0))
    if out_img.ndim == 3 and out_img.shape[0] == 3:
        out_img = np.transpose(out_img, (1, 2, 0))
    
    # Calculate metrics
    psnr = calculate_psnr(gt_img, out_img)
    ssim = calculate_ssim(gt_img, out_img)
    mae = np.mean(np.abs(gt_img - out_img))
    
    psnr_values.append(psnr)
    ssim_values.append(ssim)
    mae_values.append(mae)

# Print results
print("\n" + "=" * 60)
print("DiffCR Test Results")
print("=" * 60)
print(f"Number of samples: {len(psnr_values)}")
print(f"Average PSNR: {np.mean(psnr_values):.4f} dB")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")
print(f"Average MAE: {np.mean(mae_values):.4f}")
print("=" * 60)
