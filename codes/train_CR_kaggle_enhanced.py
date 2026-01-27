"""
Enhanced Kaggle Training Script for Cloud Removal with Cross-Attention
Uses the improved architecture with:
- Strided convolutions instead of MaxPool
- 1x1 projection after cross-attention
- Multi-component loss (L1 + SSIM + Gradient + Wavelet)

Optimized for 10-epoch quick training on Kaggle
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
import json
import math
from datetime import datetime
import time
from tqdm import tqdm
import csv

# Enable cuDNN autotuner for optimal performance
torch.backends.cudnn.benchmark = True

# Fix PyTorch 2.6 UnpicklingError
import torch.serialization
import argparse as _argparse
torch.serialization.add_safe_globals([_argparse.Namespace])

from dataloader import *
from metrics import *
from net_CR_CrossAttention import CloudRemovalCrossAttention
from loss_functions import get_cloud_removal_loss

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser = argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=8, help='batch size (increased for efficiency)')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')

parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=256, help='use full 256x256 patches')
parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/image1', help='Path to S1 and S2 data')
parser.add_argument('--data_list_filepath', type=str, default='/kaggle/input/image2/data.csv', help='Path to data.csv')
parser.add_argument('--is_use_cloudmask', type=bool, default=False)
parser.add_argument('--cloud_threshold', type=float, default=0.2)
parser.add_argument('--is_test', type=bool, default=False, help='whether in test mode')

# Loss function weights (optimized for cloud removal)
parser.add_argument('--l1_weight', type=float, default=0.5, help='L1 loss weight')
parser.add_argument('--ssim_weight', type=float, default=0.2, help='SSIM loss weight')
parser.add_argument('--gradient_weight', type=float, default=0.2, help='Gradient loss weight')
parser.add_argument('--wavelet_weight', type=float, default=0.1, help='Wavelet loss weight')

# Optimizer settings
parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW optimizer for better generalization')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for regularization')
parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['step', 'plateau', 'cosine'], help='learning rate scheduler')
parser.add_argument('--lr_patience', type=int, default=3, help='patience for plateau scheduler')
parser.add_argument('--lr_factor', type=float, default=0.5, help='factor to reduce lr')
parser.add_argument('--max_epochs', type=int, default=10, help='maximum training epochs')
parser.add_argument('--early_stop_patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping threshold')
parser.add_argument('--warmup_epochs', type=int, default=1, help='number of warmup epochs')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint every N epochs')
parser.add_argument('--save_model_dir', type=str, default='/kaggle/working/checkpoints', help='checkpoint directory')

parser.add_argument('--resume_checkpoint', type=str, default=None, help='path to resume checkpoint')
parser.add_argument('--experiment_name', type=str, default='enhanced_crossattn_10ep', help='experiment name')
parser.add_argument('--notes', type=str, default='Enhanced architecture with strided conv + 1x1 proj + multi-loss', help='notes')

# Mixed precision training
parser.add_argument('--use_amp', action='store_true', default=True, help='Use automatic mixed precision')

opts = parser.parse_args()

##===================================================##
##************** Training functions *****************##
##===================================================##
def validate(model, val_dataloader, device, criterion):
    """Validate model on validation set"""
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        try:
            val_len = len(val_dataloader)
        except Exception:
            val_len = 0

        if val_len == 0:
            print("Warning: validation set is empty. Skipping validation.")
            model.train()
            return 0.0, 0.0, 0.0

        progress_bar = tqdm(val_dataloader, desc="Validating", unit="batch")
        
        for data in progress_bar:
            try:
                # Handle both old and new key names
                cloudy_key = 'cloudy_optical' if 'cloudy_optical' in data else 'cloudy_data'
                sar_key = 'sar' if 'sar' in data else 'SAR_data'
                cloudfree_key = 'cloudfree_optical' if 'cloudfree_optical' in data else 'cloudfree_data'
                
                cloudy_optical = data[cloudy_key].to(device)
                sar_img = data[sar_key].to(device)
                cloudfree_data = data[cloudfree_key].to(device)
                
                # Forward pass
                pred = model(cloudy_optical, sar_img)
                
                # Compute loss
                loss, _ = criterion(pred, cloudfree_data)
                
                # Ensure tensors are valid
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    continue
                
                batch_psnr = PSNR(pred, cloudfree_data)
                batch_ssim = SSIM(pred, cloudfree_data)
                
                # Check if metrics are valid
                if math.isnan(batch_psnr) or math.isinf(batch_psnr) or batch_psnr <= 0:
                    continue
                if isinstance(batch_ssim, torch.Tensor):
                    batch_ssim = float(batch_ssim.item())
                if math.isnan(batch_ssim) or math.isinf(batch_ssim):
                    continue
                
                total_psnr += batch_psnr
                total_ssim += batch_ssim
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                if num_batches > 0:
                    avg_psnr = total_psnr / num_batches
                    avg_ssim = total_ssim / num_batches
                    progress_bar.set_postfix({'PSNR': f'{avg_psnr:.2f}', 'SSIM': f'{avg_ssim:.4f}'})
            
            except Exception as e:
                continue
    
    if num_batches == 0:
        print("Warning: No valid batches in validation set")
        model.train()
        return 0.0, 0.0, 0.0
    
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_loss = total_loss / num_batches
    
    model.train()
    return avg_psnr, avg_ssim, avg_loss

def save_checkpoint(model, optimizer, scheduler, epoch, val_psnr, best_val_psnr, opts):
    """Save training checkpoint"""
    os.makedirs(opts.save_model_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_psnr': val_psnr,
        'best_val_psnr': best_val_psnr,
        'opts': vars(opts)
    }
    
    # Save epoch checkpoint
    checkpoint_path = os.path.join(opts.save_model_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    is_best = val_psnr > best_val_psnr
    if is_best:
        best_path = os.path.join(opts.save_model_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model: {best_path} (PSNR: {val_psnr:.2f} dB)")
    
    return is_best

##===================================================##
##******************** Main *************************##
##===================================================##
if __name__ == '__main__':
    ##===================================================##
    ##*************** Print configuration ***************##
    ##===================================================##
    print("\n" + "="*60)
    print("Enhanced Cloud Removal Training - 10 Epochs")
    print("="*60)
    print("Architecture Improvements:")
    print("  ✓ Strided convolutions (no MaxPool)")
    print("  ✓ 1x1 projection after cross-attention")
    print("  ✓ Multi-component loss (L1+SSIM+Gradient+Wavelet)")
    print("="*60)
    for arg in vars(opts):
        print(f"{arg:.<30} {getattr(opts, arg)}")
    print("="*60 + "\n")

    ##===================================================##
    ##*************** Setup device **********************##
    ##===================================================##
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    ##===================================================##
    ##*************** Create dataloader *****************##
    ##===================================================##
    seed_torch()

    # Load train/val/test filelists
    def read_csv_rows(path):
        rows = []
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for r in reader:
                    if len(r) == 0:
                        continue
                    rows.append(r)
        except Exception:
            return []
        return rows

    train_filelist, val_filelist, test_filelist = [], [], []
    basename = os.path.basename(opts.data_list_filepath).lower()
    parent_dir = os.path.dirname(opts.data_list_filepath)

    if 'data.csv' in basename:
        train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    else:
        if 'train' in basename:
            train_filelist = read_csv_rows(opts.data_list_filepath)
            val_path = os.path.join(parent_dir, 'val.csv')
            test_path = os.path.join(parent_dir, 'test.csv')
            if os.path.exists(val_path):
                val_filelist = read_csv_rows(val_path)
            if os.path.exists(test_path):
                test_filelist = read_csv_rows(test_path)

    print(f"Training samples: {len(train_filelist)}")
    print(f"Validation samples: {len(val_filelist)}\n")

    # Training dataloader
    train_data = AlignedDataset(opts, train_filelist)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=opts.batch_sz,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True if opts.num_workers > 0 else False,
        prefetch_factor=2 if opts.num_workers > 0 else None
    )

    # Validation dataloader
    val_data = AlignedDataset(opts, val_filelist)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=opts.batch_sz,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True if opts.num_workers > 0 else False,
        prefetch_factor=2 if opts.num_workers > 0 else None
    )

    ##===================================================##
    ##****************** Create model *******************##
    ##===================================================##
    print("Creating enhanced CloudRemovalCrossAttention model...")
    model = CloudRemovalCrossAttention(
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M\n")

    ##===================================================##
    ##************** Create loss & optimizer ************##
    ##===================================================##
    print("Creating multi-component loss function...")
    criterion = get_cloud_removal_loss(
        l1_weight=opts.l1_weight,
        ssim_weight=opts.ssim_weight,
        gradient_weight=opts.gradient_weight,
        wavelet_weight=opts.wavelet_weight,
        num_channels=13
    )
    print(f"Loss weights: L1={opts.l1_weight}, SSIM={opts.ssim_weight}, "
          f"Gradient={opts.gradient_weight}, Wavelet={opts.wavelet_weight}\n")

    # Optimizer
    if opts.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opts.lr,
            weight_decay=opts.weight_decay,
            betas=(0.9, 0.999)
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opts.lr,
            weight_decay=opts.weight_decay
        )
    
    print(f"Optimizer: {opts.optimizer} (lr={opts.lr}, weight_decay={opts.weight_decay})")

    # Learning rate scheduler
    if opts.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=opts.max_epochs,
            eta_min=1e-7
        )
    elif opts.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=opts.lr_factor,
            patience=opts.lr_patience
        )
    else:  # step
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=opts.lr_factor
        )
    
    print(f"LR Scheduler: {opts.lr_scheduler}\n")

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=opts.use_amp)
    if opts.use_amp:
        print("Automatic Mixed Precision (AMP) enabled\n")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_psnr = 0.0
    epochs_without_improvement = 0

    if opts.resume_checkpoint and os.path.exists(opts.resume_checkpoint):
        print(f"Resuming from checkpoint: {opts.resume_checkpoint}")
        checkpoint = torch.load(opts.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_psnr = checkpoint.get('best_val_psnr', 0.0)
        print(f"Resumed from epoch {start_epoch}, best val PSNR: {best_val_psnr:.2f} dB\n")

    ##===================================================##
    ##**************** Train the network ****************##
    ##===================================================##
    print("="*60)
    print("Starting Training")
    print("="*60 + "\n")

    # Initialize logging
    training_log = {
        'experiment_name': opts.experiment_name,
        'notes': opts.notes,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': vars(opts),
        'epochs': []
    }
    
    log_dir = os.path.join(opts.save_model_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{opts.experiment_name}_log.json')

    train_start_time = time.time()

    # Learning rate warmup function
    def get_warmup_lr(epoch, warmup_epochs, base_lr):
        if warmup_epochs == 0 or epoch >= warmup_epochs:
            return base_lr
        return base_lr * (epoch + 1) / warmup_epochs

    for epoch in range(start_epoch, opts.max_epochs):
        # Apply warmup if needed
        if epoch < opts.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, opts.warmup_epochs, opts.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup: Setting LR to {warmup_lr:.6f}")
        
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{opts.max_epochs}")
        print(f"{'='*60}")
        
        model.train()
        
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_ssim = 0.0
        epoch_gradient = 0.0
        epoch_wavelet = 0.0
        total_train_psnr = 0.0
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Training Epoch {epoch+1}",
            unit="batch"
        )
        
        for batch_idx, data in enumerate(progress_bar):
            try:
                # Handle both old and new key names
                cloudy_key = 'cloudy_optical' if 'cloudy_optical' in data else 'cloudy_data'
                sar_key = 'sar' if 'sar' in data else 'SAR_data'
                cloudfree_key = 'cloudfree_optical' if 'cloudfree_optical' in data else 'cloudfree_data'
                
                cloudy_optical = data[cloudy_key].to(device)
                sar_img = data[sar_key].to(device)
                cloudfree_data = data[cloudfree_key].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with AMP
                with torch.cuda.amp.autocast(enabled=opts.use_amp):
                    pred = model(cloudy_optical, sar_img)
                    loss, loss_dict = criterion(pred, cloudfree_data)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if opts.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                
                # Accumulate losses
                epoch_loss += loss_dict['total']
                epoch_l1 += loss_dict['l1']
                epoch_ssim += loss_dict['ssim']
                epoch_gradient += loss_dict['gradient']
                epoch_wavelet += loss_dict['wavelet']
                
                # Calculate batch PSNR
                with torch.no_grad():
                    batch_psnr = PSNR(pred.float(), cloudfree_data.float())
                    if isinstance(batch_psnr, torch.Tensor):
                        batch_psnr = float(batch_psnr.item())
                    if not (math.isnan(batch_psnr) or math.isinf(batch_psnr)):
                        total_train_psnr += batch_psnr
                
                num_batches += 1
                
                # Update progress bar
                avg_loss = epoch_loss / num_batches
                avg_psnr = total_train_psnr / num_batches
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'PSNR': f'{avg_psnr:.2f}'
                })
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Epoch statistics
        if num_batches > 0:
            avg_train_loss = epoch_loss / num_batches
            avg_train_psnr = total_train_psnr / num_batches
            avg_l1 = epoch_l1 / num_batches
            avg_ssim_loss = epoch_ssim / num_batches
            avg_grad = epoch_gradient / num_batches
            avg_wave = epoch_wavelet / num_batches
        else:
            avg_train_loss = avg_train_psnr = avg_l1 = avg_ssim_loss = avg_grad = avg_wave = 0.0
        
        # Validation
        print(f"\nRunning validation...")
        val_psnr, val_ssim, val_loss = validate(model, val_dataloader, device, criterion)
        
        # Check if best model
        is_best = val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_psnr
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Update learning rate
        if epoch >= opts.warmup_epochs:
            if opts.lr_scheduler == 'plateau':
                scheduler.step(val_psnr)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_psnr, best_val_psnr, opts)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch results
        epoch_log = {
            'epoch': epoch,
            'train_loss': float(avg_train_loss),
            'train_psnr': float(avg_train_psnr),
            'train_l1': float(avg_l1),
            'train_ssim': float(avg_ssim_loss),
            'train_gradient': float(avg_grad),
            'train_wavelet': float(avg_wave),
            'val_loss': float(val_loss),
            'val_psnr': float(val_psnr),
            'val_ssim': float(val_ssim),
            'best_val_psnr': float(best_val_psnr),
            'learning_rate': float(current_lr),
            'epoch_time': float(epoch_time),
            'is_best': is_best,
            'epochs_without_improvement': epochs_without_improvement
        }
        training_log['epochs'].append(epoch_log)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} (L1:{avg_l1:.4f} SSIM:{avg_ssim_loss:.4f} Grad:{avg_grad:.4f} Wave:{avg_wave:.4f})")
        print(f"  Train PSNR: {avg_train_psnr:.2f} dB")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val PSNR:   {val_psnr:.2f} dB")
        print(f"  Val SSIM:   {val_ssim:.4f}")
        print(f"  Best Val PSNR: {best_val_psnr:.2f} dB {'(NEW!)' if is_best else ''}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Epochs without improvement: {epochs_without_improvement}")
        print(f"  Epoch Time: {epoch_time/60:.1f} minutes")
        print(f"{'='*60}")
        
        # Early stopping check
        if opts.early_stop_patience > 0 and epochs_without_improvement >= opts.early_stop_patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered!")
            print(f"No improvement for {epochs_without_improvement} epochs")
            print(f"Best validation PSNR: {best_val_psnr:.2f} dB")
            print(f"{'='*60}\n")
            break
        
        # Save training log
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

    # Training complete
    total_time = time.time() - train_start_time
    training_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_log['total_time_hours'] = total_time / 3600

    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Training log saved to: {log_path}")
    best_model_path = os.path.join(opts.save_model_dir, 'best_model.pth')
    print(f"Best model saved to: {best_model_path}")
    print(f"{'='*60}\n")
    
    # Copy checkpoints to Kaggle output
    try:
        import shutil
        output_dir = '/kaggle/working'
        
        # Copy best model
        if os.path.exists(best_model_path):
            output_best = os.path.join(output_dir, 'best_model.pth')
            shutil.copy(best_model_path, output_best)
            print(f"✅ Best model copied to: {output_best}")
        
        # Copy training log
        output_log = os.path.join(output_dir, 'training_log.json')
        shutil.copy(log_path, output_log)
        print(f"✅ Training log copied to: {output_log}")
        
        print(f"\n{'='*60}")
        print("Checkpoints ready for download from Kaggle!")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Note: Could not copy to output (not on Kaggle?): {e}")
