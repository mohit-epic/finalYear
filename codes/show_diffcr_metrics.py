#!/usr/bin/env python3
"""
Extract and display DiffCR test metrics
Run this after testing to see PSNR, SSIM, and MAE results
"""

import os
import json
import glob

# Find the latest experiment directory
exp_base = "/kaggle/working/finalYear/codes/DiffCR/experiments"
exp_dir = os.path.join(exp_base, "sen12ms_cr_diffcr_test")

print("=" * 60)
print("DiffCR Test Results")
print("=" * 60)

# Try to find metrics in log files or saved results
log_files = glob.glob(os.path.join(exp_dir, "**/*.log"), recursive=True)
json_files = glob.glob(os.path.join(exp_dir, "**/*.json"), recursive=True)

# Look for metrics in log files
metrics_found = False
for log_file in log_files:
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            if 'mae' in content.lower() or 'psnr' in content.lower():
                print(f"\nFrom {os.path.basename(log_file)}:")
                # Extract metric lines
                for line in content.split('\n'):
                    if any(metric in line.lower() for metric in ['mae', 'psnr', 'ssim']):
                        print(f"  {line.strip()}")
                        metrics_found = True
    except:
        pass

# Look for metrics in JSON files
for json_file in json_files:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                print(f"\nFrom {os.path.basename(json_file)}:")
                for key in ['mae', 'psnr', 'ssim', 'MAE', 'PSNR', 'SSIM']:
                    if key in data or key.lower() in data:
                        val = data.get(key) or data.get(key.lower())
                        print(f"  {key.upper()}: {val}")
                        metrics_found = True
    except:
        pass

if not metrics_found:
    print("\n⚠️  Metrics not found in log files.")
    print("\nTo calculate metrics manually, run:")
    print("  python calculate_metrics.py")
    print("\nOr check the experiment directory:")
    print(f"  {exp_dir}")

print("\n" + "=" * 60)
