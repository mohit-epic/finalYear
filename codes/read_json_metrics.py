#!/usr/bin/env python3
"""
Read and display metrics from DiffCR JSON results
"""

import json
import os
import glob

# Find JSON files in results directory
results_dir = "/kaggle/working/diffcr_results"

json_files = glob.glob(os.path.join(results_dir, "*.json"))

if not json_files:
    print(f"No JSON files found in {results_dir}")
    print("\nListing directory contents:")
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            print(f"  {item}")
else:
    print("=" * 60)
    print("DiffCR Test Results")
    print("=" * 60)
    
    for json_file in json_files:
        print(f"\nFrom: {os.path.basename(json_file)}")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Print all metrics found
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value}")
                    elif key.lower() in ['psnr', 'ssim', 'mae']:
                        print(f"  {key}: {value}")
            elif isinstance(data, list):
                print(f"  Found {len(data)} entries")
                # Show first entry as example
                if len(data) > 0:
                    print(f"  Sample entry: {data[0]}")
        except Exception as e:
            print(f"  Error reading file: {e}")
    
    print("=" * 60)
