"""
Test Script for diffCR (CloudRemovalCrossAttention) Model on Kaggle
Tests the pretrained model on the test dataset split
Computes: PSNR, SSIM, SAM, RMSE

Usage on Kaggle (copy-paste this command):
python test_diffCR_kaggle.py --checkpoint_path /kaggle/input/your-model-dataset/best_model.pth --data_list_filepath /kaggle/input/your-data-csv/data.csv
"""

import os
import torch
import argparse
import json
from datetime import datetime
from tqdm import tqdm

# Fix PyTorch 2.6 UnpicklingError
import torch.serialization
import argparse as _argparse
torch.serialization.add_safe_globals([_argparse.Namespace])

from dataloader import AlignedDataset, get_train_val_test_filelists
from metrics import PSNR, SSIM, SAM, RMSE
from net_CR_CrossAttention import CloudRemovalCrossAttention

##===================================================##
##********** Configure test settings ****************##
##===================================================##
parser = argparse.ArgumentParser(description='Test diffCR Model')

# Data settings
parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/sen12ms-cr-winter')
parser.add_argument('--data_list_filepath', type=str, default='/kaggle/working/data.csv')
parser.add_argument('--is_use_cloudmask', type=bool, default=False)
parser.add_argument('--cloud_threshold', type=float, default=0.2)
parser.add_argument('--is_test', type=bool, default=True)

# Model settings
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to pretrained checkpoint')
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--attn_drop', type=float, default=0.1)
parser.add_argument('--proj_drop', type=float, default=0.1)

opts = parser.parse_args()

##===================================================##
##****************** Main Test **********************##
##===================================================##
def main():
    print("\n" + "="*60)
    print("Testing diffCR (CloudRemovalCrossAttention) Model")
    print("="*60)
    print(f"Checkpoint: {opts.checkpoint_path}")
    print(f"Data CSV: {opts.data_list_filepath}")
    print("="*60 + "\n")
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Load test data
    print("Loading test dataset...")
    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    print(f"Test samples: {len(test_filelist)}\n")
    
    test_data = AlignedDataset(opts, test_filelist)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print("Creating CloudRemovalCrossAttention model...")
    model = CloudRemovalCrossAttention(
        num_heads=opts.num_heads,
        qkv_bias=True,
        attn_drop=opts.attn_drop,
        proj_drop=opts.proj_drop
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {opts.checkpoint_path}")
    checkpoint = torch.load(opts.checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        if 'val_psnr' in checkpoint:
            print(f"Validation PSNR: {checkpoint['val_psnr']:.2f} dB")
    else:
        state_dict = checkpoint
    
    # Handle DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print("✓ Model loaded successfully\n")
    
    # Test
    print("="*60)
    print("Testing on test set...")
    print("="*60 + "\n")
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_sam = 0.0
    total_rmse = 0.0
    num_samples = 0
    results_per_image = []
    
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Testing"):
            cloudy_data = data['cloudy_data'].to(device)
            cloudfree_data = data['cloudfree_data'].to(device)
            sar_data = data['SAR_data'].to(device)
            file_name = data['file_name'][0] if isinstance(data['file_name'], list) else data['file_name']
            
            # Forward pass
            pred = model(cloudy_data, sar_data)
            
            # Compute metrics
            psnr_val = PSNR(pred, cloudfree_data)
            ssim_val = SSIM(pred, cloudfree_data)
            sam_val = SAM(pred, cloudfree_data)
            rmse_val = RMSE(pred, cloudfree_data)
            
            # Convert to float
            psnr_val = float(psnr_val) if not isinstance(psnr_val, float) else psnr_val
            ssim_val = float(ssim_val.item()) if hasattr(ssim_val, 'item') else float(ssim_val)
            sam_val = float(sam_val.item()) if hasattr(sam_val, 'item') else float(sam_val)
            rmse_val = float(rmse_val.item()) if hasattr(rmse_val, 'item') else float(rmse_val)
            
            total_psnr += psnr_val
            total_ssim += ssim_val
            total_sam += sam_val
            total_rmse += rmse_val
            num_samples += 1
            
            results_per_image.append({
                'image': file_name,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'sam': sam_val,
                'rmse': rmse_val
            })
    
    # Compute averages
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_sam = total_sam / num_samples
    avg_rmse = total_rmse / num_samples
    
    # Print results
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Number of samples: {num_samples}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average SAM:  {avg_sam:.4f} degrees")
    print(f"Average RMSE: {avg_rmse:.5f}")
    print("="*60 + "\n")
    
    # Save results
    results_dir = '/kaggle/working/results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'diffCR_test_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'model': 'CloudRemovalCrossAttention (diffCR)',
            'checkpoint': opts.checkpoint_path,
            'num_samples': num_samples,
            'average_metrics': {
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'sam': avg_sam,
                'rmse': avg_rmse
            },
            'per_image_results': results_per_image
        }, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}\n")

if __name__ == '__main__':
    main()
