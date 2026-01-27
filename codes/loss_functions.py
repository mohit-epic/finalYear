"""
Enhanced Loss Functions for Cloud Removal
Combines multiple loss components for better texture and structure preservation:
1. L1 Loss - Pixel-wise accuracy
2. SSIM Loss - Structural similarity
3. Gradient Loss - Edge preservation
4. Wavelet Loss - Multi-scale frequency preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss
    Measures perceptual similarity between images
    Better for texture and structure preservation than L1/L2
    """
    def __init__(self, window_size=11, size_average=True, channel=13):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)
        
    def gaussian(self, window_size, sigma=1.5):
        """Create Gaussian kernel"""
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        """Create 2D Gaussian window"""
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        (_, channel, _, _) = img1.size()
        
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device).type_as(img1)
        
        window = self.window
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        """Return SSIM loss (1 - SSIM)"""
        return 1 - self.ssim(img1, img2)


class GradientLoss(nn.Module):
    """
    Gradient Loss for edge preservation
    Computes L1 loss on image gradients (Sobel filters)
    Critical for preserving cloud boundaries and texture details
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Sobel kernels for gradient computation
        self.sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        self.sobel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        
    def gradient(self, img):
        """Compute image gradients using Sobel filters"""
        B, C, H, W = img.shape
        
        # Move kernels to same device as input
        if self.sobel_x.device != img.device:
            self.sobel_x = self.sobel_x.to(img.device).type_as(img)
            self.sobel_y = self.sobel_y.to(img.device).type_as(img)
        
        # Expand kernels for all channels
        sobel_x = self.sobel_x.repeat(C, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1)
        
        # Compute gradients
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
        
        return grad_x, grad_y
    
    def forward(self, pred, target):
        """Compute gradient loss"""
        pred_grad_x, pred_grad_y = self.gradient(pred)
        target_grad_x, target_grad_y = self.gradient(target)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


class WaveletLoss(nn.Module):
    """
    Wavelet Loss for multi-scale frequency preservation
    Uses Haar wavelet transform to decompose images into frequency bands
    Ensures both low and high frequency components are preserved
    """
    def __init__(self):
        super(WaveletLoss, self).__init__()
        
    def haar_wavelet_transform(self, img):
        """
        Perform single-level 2D Haar wavelet transform
        Returns: LL (approximation), LH, HL, HH (details)
        """
        # Average pooling for low-pass
        ll = F.avg_pool2d(img, kernel_size=2, stride=2)
        
        # Compute high-frequency components
        # LH: horizontal details
        lh = F.avg_pool2d(img[:, :, :, 0::2] - img[:, :, :, 1::2], kernel_size=(2, 1), stride=(2, 1))
        
        # HL: vertical details  
        hl = F.avg_pool2d(img[:, :, 0::2, :] - img[:, :, 1::2, :], kernel_size=(1, 2), stride=(1, 2))
        
        # HH: diagonal details
        hh_temp = img[:, :, 0::2, 0::2] - img[:, :, 1::2, 0::2] - img[:, :, 0::2, 1::2] + img[:, :, 1::2, 1::2]
        hh = F.avg_pool2d(hh_temp, kernel_size=1, stride=1)
        
        return ll, lh, hl, hh
    
    def forward(self, pred, target):
        """Compute wavelet loss on all frequency bands"""
        # Ensure even dimensions for wavelet transform
        B, C, H, W = pred.shape
        if H % 2 != 0 or W % 2 != 0:
            pred = F.pad(pred, (0, W % 2, 0, H % 2), mode='reflect')
            target = F.pad(target, (0, W % 2, 0, H % 2), mode='reflect')
        
        # Decompose both images
        pred_ll, pred_lh, pred_hl, pred_hh = self.haar_wavelet_transform(pred)
        target_ll, target_lh, target_hl, target_hh = self.haar_wavelet_transform(target)
        
        # Compute L1 loss on each band
        loss_ll = F.l1_loss(pred_ll, target_ll)
        loss_lh = F.l1_loss(pred_lh, target_lh)
        loss_hl = F.l1_loss(pred_hl, target_hl)
        loss_hh = F.l1_loss(pred_hh, target_hh)
        
        # Weight high-frequency components more (they're critical for texture)
        return loss_ll + 1.5 * (loss_lh + loss_hl + loss_hh)


class CombinedCloudRemovalLoss(nn.Module):
    """
    Combined Loss Function for Cloud Removal
    
    Weights (tunable):
    - L1: 0.5 (pixel accuracy)
    - SSIM: 0.2 (structural similarity)
    - Gradient: 0.2 (edge preservation)
    - Wavelet: 0.1 (frequency preservation)
    
    Total weight = 1.0
    """
    def __init__(self, 
                 l1_weight=0.5, 
                 ssim_weight=0.2, 
                 gradient_weight=0.2, 
                 wavelet_weight=0.1,
                 num_channels=13):
        super(CombinedCloudRemovalLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.wavelet_weight = wavelet_weight
        
        # Initialize loss components
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(channel=num_channels)
        self.gradient_loss = GradientLoss()
        self.wavelet_loss = WaveletLoss()
        
    def forward(self, pred, target):
        """
        Compute combined loss
        
        Args:
            pred: Predicted cloud-free image (B, C, H, W)
            target: Ground truth cloud-free image (B, C, H, W)
            
        Returns:
            total_loss: Weighted combination of all losses
            loss_dict: Dictionary with individual loss components (for logging)
        """
        # Compute individual losses
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        gradient = self.gradient_loss(pred, target)
        wavelet = self.wavelet_loss(pred, target)
        
        # Weighted combination
        total_loss = (
            self.l1_weight * l1 +
            self.ssim_weight * ssim +
            self.gradient_weight * gradient +
            self.wavelet_weight * wavelet
        )
        
        # Return loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1.item(),
            'ssim': ssim.item(),
            'gradient': gradient.item(),
            'wavelet': wavelet.item()
        }
        
        return total_loss, loss_dict


# Convenience function for easy import
def get_cloud_removal_loss(l1_weight=0.5, ssim_weight=0.2, 
                           gradient_weight=0.2, wavelet_weight=0.1,
                           num_channels=13):
    """
    Factory function to create the combined loss
    
    Usage in training script:
        criterion = get_cloud_removal_loss()
        loss, loss_dict = criterion(output, target)
        loss.backward()
    """
    return CombinedCloudRemovalLoss(
        l1_weight=l1_weight,
        ssim_weight=ssim_weight,
        gradient_weight=gradient_weight,
        wavelet_weight=wavelet_weight,
        num_channels=num_channels
    )


if __name__ == "__main__":
    # Test the loss functions
    print("Testing Combined Cloud Removal Loss...")
    
    batch_size = 2
    channels = 13
    height, width = 256, 256
    
    # Create dummy data
    pred = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    
    # Test combined loss
    criterion = get_cloud_removal_loss()
    loss, loss_dict = criterion(pred, target)
    
    print(f"\n✓ Loss computation successful!")
    print(f"\nLoss Components:")
    print(f"  Total Loss:    {loss_dict['total']:.4f}")
    print(f"  L1 Loss:       {loss_dict['l1']:.4f}")
    print(f"  SSIM Loss:     {loss_dict['ssim']:.4f}")
    print(f"  Gradient Loss: {loss_dict['gradient']:.4f}")
    print(f"  Wavelet Loss:  {loss_dict['wavelet']:.4f}")
    
    print(f"\n✓ All loss functions working correctly!")
    print(f"\nUsage in training:")
    print(f"  from loss_functions import get_cloud_removal_loss")
    print(f"  criterion = get_cloud_removal_loss()")
    print(f"  loss, loss_dict = criterion(output, target)")
    print(f"  loss.backward()")
