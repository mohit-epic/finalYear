import torch
import torch.utils.data as data
import numpy as np
import tifffile
import os
import csv

class SEN12MS_CR_RGB(data.Dataset):
    """
    SEN12MS-CR dataset adapted for DiffCR (RGB only)
    Extracts RGB bands from 13-band multi-spectral images
    """
    def __init__(self, data_root, mode='test', data_csv='/kaggle/working/data.csv'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        
        # Load CSV
        csv_file = open(data_csv, "r")
        list_reader = csv.reader(csv_file)
        
        for f in list_reader:
            if mode == 'train' and f[0] == '1':
                self.filepair.append(f)
            elif mode == 'val' and f[0] == '2':
                self.filepair.append(f)
            elif mode == 'test' and f[0] == '3':
                self.filepair.append(f)
        
        csv_file.close()
        
        # Limit to 50 samples for faster testing (~2 minutes instead of 4 hours)
        # Comment out the next line to test on all samples
        self.filepair = self.filepair[:50]
        
        print(f"Loaded {len(self.filepair)} {mode} samples from SEN12MS-CR")
    
    def __getitem__(self, index):
        fileID = self.filepair[index]
        
        # Parse paths based on CSV format
        if len(fileID) == 8:
            # New format with dataset_type
            dataset_type = fileID[1]
            
            if dataset_type == 'winter':
                root = '/kaggle/input/sen12ms-cr-winter'
            elif dataset_type == 'spring':
                root = '/kaggle/input/t-glf-cr-winter'
            elif dataset_type == 'fall':
                root = '/kaggle/input/t-glf-cr-fall'
            else:
                root = self.data_root
            
            s2_cloudy_path = os.path.join(root, fileID[4], fileID[7])
            s2_cloudfree_path = os.path.join(root, fileID[3], fileID[5])
            reference_filename = fileID[5]
        elif len(fileID) == 7:
            # Format without dataset_type
            s2_cloudy_path = os.path.join(self.data_root, fileID[3], fileID[6])
            s2_cloudfree_path = os.path.join(self.data_root, fileID[2], fileID[4])
            reference_filename = fileID[4]
        else:
            # Old format
            s2_cloudy_path = os.path.join(self.data_root, fileID[3], fileID[4])
            s2_cloudfree_path = os.path.join(self.data_root, fileID[2], fileID[4])
            reference_filename = fileID[4]
        
        # Load and extract RGB
        cloudy_img = self.load_rgb(s2_cloudy_path)
        cloudfree_img = self.load_rgb(s2_cloudfree_path)
        
        # DiffCR's NAFNet expects 3 cloudy images (for temporal fusion)
        # Since SEN12MS-CR has only 1 cloudy image, we concatenate 3 copies along channel dimension
        # This creates a 9-channel tensor (3 images × 3 RGB channels)
        # Shape: (9, H, W) which will be concatenated with noisy image (3, H, W) = (12, H, W) total
        cond_image_stacked = torch.cat([cloudy_img, cloudy_img, cloudy_img], dim=0)  # Shape: (9, H, W)
        
        return {
            'gt_image': cloudfree_img,
            'cond_image': cond_image_stacked,
            'path': reference_filename
        }
    
    def load_rgb(self, path):
        """
        Load TIF image and extract RGB bands
        
        Band mapping for Sentinel-2:
        - Band 4 (Red): Index 3
        - Band 3 (Green): Index 2
        - Band 2 (Blue): Index 1
        """
        # Load TIF
        img = tifffile.imread(path)
        
        # Handle different input shapes
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 3:
            h, w, c = img.shape
            # If channel-last format, transpose to channel-first
            if c <= 20 and h > c and w > c:
                img = np.transpose(img, (2, 0, 1))
        
        # Fill NaN values
        img[np.isnan(img)] = np.nanmean(img)
        
        # Extract RGB bands [B4, B3, B2] = indices [3, 2, 1]
        rgb = img[[3, 2, 1], :, :]
        
        # Normalize to [0, 1]
        rgb = np.clip(rgb, 0, 10000) / 10000.0
        
        # Convert to tensor
        image = torch.from_numpy(rgb.copy()).float()
        
        # Normalize to [-1, 1] for DiffCR
        # DiffCR uses: (x - 0.5) / 0.5
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        
        image.sub_(mean).div_(std)
        
        return image
    
    def __len__(self):
        return len(self.filepair)
