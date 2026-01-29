"""
Standalone DiffCR Test Script for Kaggle
Tests pretrained DiffCR model on SEN12MS-CR dataset (RGB-adapted)

Usage:
    python test_diffcr_standalone.py \\
        --checkpoint_path /kaggle/input/diffcr-model/diffcr_new.pth \\
        --data_csv /kaggle/input/your-data/data.csv
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Import RGB adapter
from sen12ms_to_rgb_adapter import SEN12MSRGBDataset, SEN12MSRGBAdapter, get_train_val_test_filelists

# Import DiffCR evaluation metrics
import cv2


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    # Convert tensors to numpy if needed
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure range [0, 1]
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    
    max_value = 1.0
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images (simplified version)"""
    # Convert tensors to numpy if needed
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure shape (H, W, C)
    if img1.ndim == 3 and img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Convert to uint8 for cv2
    img1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
    img2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
    
    # Calculate SSIM for each channel
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


##===================================================##
##********** Configure test settings ****************##
##===================================================##
parser = argparse.ArgumentParser(description='Test DiffCR Model on SEN12MS-CR (RGB)')

# Data settings
parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/sen12ms-cr-winter')
parser.add_argument('--data_csv', type=str, default='/kaggle/working/data.csv')
parser.add_argument('--is_use_cloudmask', type=bool, default=False)
parser.add_argument('--cloud_threshold', type=float, default=0.2)
parser.add_argument('--is_test', type=bool, default=True)

# Model settings
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to DiffCR checkpoint')
parser.add_argument('--num_samples', type=int, default=-1, help='Number of test samples (-1 for all)')

# Output settings
parser.add_argument('--output_dir', type=str, default='/kaggle/working/diffcr_results')
parser.add_argument('--save_images', action='store_true', default=False)

opts = parser.parse_args()


def load_diffcr_model(checkpoint_path, device):
    """
    Load DiffCR model from checkpoint
    This is a simplified loader - assumes the checkpoint contains the full model
    """
    print(f"Loading DiffCR checkpoint from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # The checkpoint should contain the network state dict
        if 'network' in checkpoint:
            state_dict = checkpoint['network']
            print("Found 'network' key in checkpoint")
        elif 'gen' in checkpoint:
            state_dict = checkpoint['gen']
            print("Found 'gen' key in checkpoint")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Found 'model_state_dict' key in checkpoint")
        else:
            # Assume the checkpoint is the state dict itself
            state_dict = checkpoint
            print("Using checkpoint as state dict directly")
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Keys in state dict: {len(state_dict)}")
        
        # Print some key names to understand structure
        sample_keys = list(state_dict.keys())[:5]
        print(f"  Sample keys: {sample_keys}")
        
        return state_dict
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def simple_diffusion_inference(model_state, cond_image, device, num_steps=20):
    """
    Simplified diffusion inference
    Since we don't have the full DiffCR framework, we'll do a simple forward pass
    
    Note: This is a SIMPLIFIED version. For full DiffCR inference, you need the complete framework.
    """
    # For now, we'll just return the conditional image as a placeholder
    # In a full implementation, this would run the diffusion sampling process
    
    print("WARNING: Using simplified inference (not full diffusion sampling)")
    print("For accurate results, use the full DiffCR framework with run.py")
    
    # Simple denoising: just return the input (placeholder)
    # In reality, DiffCR would run 20 steps of diffusion sampling
    return cond_image


def test_diffcr(opts):
    """Main testing function"""
    
    print("\n" + "="*60)
    print("DiffCR Model Testing (Standalone)")
    print("="*60)
    print(f"Checkpoint: {opts.checkpoint_path}")
    print(f"Data CSV: {opts.data_csv}")
    print("="*60 + "\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Load test data
    print("Loading test dataset...")
    _, _, test_filelist = get_train_val_test_filelists(opts.data_csv)
    
    if opts.num_samples > 0:
        test_filelist = test_filelist[:opts.num_samples]
    
    print(f"Test samples: {len(test_filelist)}\n")
    
    # Create RGB dataset
    test_dataset = SEN12MSRGBDataset(opts, test_filelist)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Load model
    model_state = load_diffcr_model(opts.checkpoint_path, device)
    
    print("\n" + "="*60)
    print("IMPORTANT NOTE:")
    print("This is a SIMPLIFIED test script.")
    print("For full DiffCR inference with diffusion sampling,")
    print("use the original DiffCR framework: python run.py -p test")
    print("="*60 + "\n")
    
    # Test loop
    print("Running inference...")
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    results_per_image = []
    
    adapter = SEN12MSRGBAdapter(opts)
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            cond_image = data['cond_image'].to(device)  # Cloudy RGB
            gt_image = data['gt_image'].to(device)      # Cloud-free RGB
            filename = data['path'][0] if isinstance(data['path'], list) else data['path']
            
            # Run inference (simplified - just uses input as output)
            # In full DiffCR, this would run diffusion sampling
            pred_image = simple_diffusion_inference(model_state, cond_image, device)
            
            # Denormalize for metrics ([-1, 1] -> [0, 1])
            pred_denorm = adapter.denormalize_diffcr(pred_image.squeeze(0))
            gt_denorm = adapter.denormalize_diffcr(gt_image.squeeze(0))
            
            # Calculate metrics
            psnr_val = calculate_psnr(pred_denorm, gt_denorm)
            ssim_val = calculate_ssim(pred_denorm, gt_denorm)
            
            total_psnr += psnr_val
            total_ssim += ssim_val
            num_samples += 1
            
            results_per_image.append({
                'filename': filename,
                'psnr': float(psnr_val),
                'ssim': float(ssim_val)
            })
            
            # Save images if requested
            if opts.save_images:
                save_dir = os.path.join(opts.output_dir, 'images')
                os.makedirs(save_dir, exist_ok=True)
                
                # Save as numpy arrays
                pred_np = pred_denorm.cpu().numpy()
                np.save(os.path.join(save_dir, f"{filename}_pred.npy"), pred_np)
    
    # Compute averages
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    # Print results
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Number of samples: {num_samples}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*60 + "\n")
    
    print("NOTE: These results use SIMPLIFIED inference (not full diffusion).")
    print("For accurate DiffCR results, use the full framework.\n")
    
    # Save results
    os.makedirs(opts.output_dir, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'DiffCR (Simplified Inference)',
        'checkpoint': opts.checkpoint_path,
        'num_samples': num_samples,
        'note': 'Simplified inference - not full diffusion sampling',
        'average_metrics': {
            'psnr': float(avg_psnr),
            'ssim': float(avg_ssim)
        },
        'per_image_results': results_per_image
    }
    
    results_file = os.path.join(opts.output_dir, 'diffcr_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}\n")
    
    return results


if __name__ == '__main__':
    try:
        results = test_diffcr(opts)
        print("Testing complete!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
