"""
SEN12MS-CR to RGB Adapter for DiffCR Model
Extracts RGB bands from 13-band multi-spectral images and applies DiffCR normalization
"""

import os
import numpy as np
import torch
import tifffile
import csv


class SEN12MSRGBAdapter:
    """
    Adapter to convert SEN12MS-CR multi-spectral data to RGB format for DiffCR
    
    Band mapping for Sentinel-2:
    - Band 4 (Red): Index 3
    - Band 3 (Green): Index 2  
    - Band 2 (Blue): Index 1
    """
    
    def __init__(self, opts):
        self.opts = opts
        # RGB band indices in 13-band Sentinel-2 data
        # Bands: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]
        # RGB = [B4, B3, B2] = indices [3, 2, 1]
        self.rgb_indices = [3, 2, 1]  # R, G, B
        
    def load_tif_image(self, path):
        """Load TIF image and ensure proper shape (C, H, W)"""
        image = tifffile.imread(path)
        
        # Handle different input shapes
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:
            h, w, c = image.shape
            # If channel-last format, transpose to channel-first
            if c <= 20 and h > c and w > c:
                image = np.transpose(image, (2, 0, 1))
        
        # Fill NaN values
        image[np.isnan(image)] = np.nanmean(image)
        return image
    
    def extract_rgb(self, multispectral_image):
        """
        Extract RGB bands from 13-band multi-spectral image
        
        Args:
            multispectral_image: (13, H, W) numpy array
            
        Returns:
            rgb_image: (3, H, W) numpy array
        """
        if multispectral_image.shape[0] < 4:
            raise ValueError(f"Expected at least 4 bands, got {multispectral_image.shape[0]}")
        
        # Extract RGB bands
        rgb_image = multispectral_image[self.rgb_indices, :, :]
        return rgb_image
    
    def normalize_diffcr(self, image):
        """
        Apply DiffCR normalization: (x / 10000.0 - 0.5) / 0.5
        This maps [0, 10000] -> [-1, 1]
        
        Args:
            image: (C, H, W) numpy array
            
        Returns:
            normalized: (C, H, W) torch tensor in range [-1, 1]
        """
        # Clip to valid range
        image = np.clip(image, 0, 10000)
        
        # Normalize to [0, 1]
        image = image / 10000.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image.copy()).float()
        
        # Normalize to [-1, 1]: (x - 0.5) / 0.5
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=image_tensor.dtype, device=image_tensor.device)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=image_tensor.dtype, device=image_tensor.device)
        
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        
        image_tensor.sub_(mean).div_(std)
        
        return image_tensor
    
    def denormalize_diffcr(self, image_tensor):
        """
        Reverse DiffCR normalization: x * 0.5 + 0.5
        Maps [-1, 1] -> [0, 1]
        
        Args:
            image_tensor: (C, H, W) torch tensor in range [-1, 1]
            
        Returns:
            denormalized: (C, H, W) torch tensor in range [0, 1]
        """
        return image_tensor * 0.5 + 0.5
    
    def load_and_prepare_rgb(self, cloudy_path, cloudfree_path):
        """
        Load multi-spectral images and prepare RGB versions for DiffCR
        
        Args:
            cloudy_path: Path to cloudy multi-spectral image
            cloudfree_path: Path to cloud-free multi-spectral image
            
        Returns:
            cloudy_rgb: (3, H, W) tensor, normalized to [-1, 1]
            cloudfree_rgb: (3, H, W) tensor, normalized to [-1, 1]
        """
        # Load multi-spectral images
        cloudy_ms = self.load_tif_image(cloudy_path)
        cloudfree_ms = self.load_tif_image(cloudfree_path)
        
        # Extract RGB bands
        cloudy_rgb = self.extract_rgb(cloudy_ms)
        cloudfree_rgb = self.extract_rgb(cloudfree_ms)
        
        # Apply DiffCR normalization
        cloudy_rgb_norm = self.normalize_diffcr(cloudy_rgb)
        cloudfree_rgb_norm = self.normalize_diffcr(cloudfree_rgb)
        
        return cloudy_rgb_norm, cloudfree_rgb_norm


class SEN12MSRGBDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for SEN12MS-CR data with RGB extraction
    Compatible with DiffCR model expectations
    """
    
    def __init__(self, opts, filelist):
        self.opts = opts
        self.filelist = filelist
        self.adapter = SEN12MSRGBAdapter(opts)
        
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        fileID = self.filelist[index]
        
        # Parse file paths (same logic as original dataloader)
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
                root = self.opts.input_data_folder
            
            s2_cloudy_path = os.path.join(root, fileID[4], fileID[7])
            s2_cloudfree_path = os.path.join(root, fileID[3], fileID[5])
            reference_filename = fileID[5]
        elif len(fileID) == 7:
            # Format without dataset_type
            s2_cloudy_path = os.path.join(self.opts.input_data_folder, fileID[3], fileID[6])
            s2_cloudfree_path = os.path.join(self.opts.input_data_folder, fileID[2], fileID[4])
            reference_filename = fileID[4]
        else:
            # Old format
            s2_cloudy_path = os.path.join(self.opts.input_data_folder, fileID[3], fileID[4])
            s2_cloudfree_path = os.path.join(self.opts.input_data_folder, fileID[2], fileID[4])
            reference_filename = fileID[4]
        
        # Load and prepare RGB data
        cloudy_rgb, cloudfree_rgb = self.adapter.load_and_prepare_rgb(
            s2_cloudy_path, s2_cloudfree_path
        )
        
        # Return in DiffCR format
        return {
            'cond_image': cloudy_rgb,      # Cloudy RGB (input)
            'gt_image': cloudfree_rgb,     # Cloud-free RGB (ground truth)
            'path': reference_filename
        }


def get_train_val_test_filelists(listpath):
    """Read data.csv and split into train/val/test"""
    csv_file = open(listpath, "r")
    list_reader = csv.reader(csv_file)
    
    train_filelist = []
    val_filelist = []
    test_filelist = []
    
    for f in list_reader:
        line_entries = f
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)
    
    csv_file.close()
    return train_filelist, val_filelist, test_filelist


if __name__ == '__main__':
    """Test the RGB adapter"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/sen12ms-cr-winter')
    parser.add_argument('--data_list_filepath', type=str, default='/kaggle/working/data.csv')
    opts = parser.parse_args()
    
    print("Testing SEN12MS RGB Adapter...")
    
    # Test adapter
    adapter = SEN12MSRGBAdapter(opts)
    
    # Load test filelists
    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    
    if len(test_filelist) > 0:
        print(f"Found {len(test_filelist)} test samples")
        
        # Create dataset
        dataset = SEN12MSRGBDataset(opts, test_filelist[:5])
        
        # Test loading
        for i, sample in enumerate(dataset):
            print(f"\nSample {i}:")
            print(f"  Cloudy RGB shape: {sample['cond_image'].shape}")
            print(f"  Cloud-free RGB shape: {sample['gt_image'].shape}")
            print(f"  Cloudy RGB range: [{sample['cond_image'].min():.3f}, {sample['cond_image'].max():.3f}]")
            print(f"  Cloud-free RGB range: [{sample['gt_image'].min():.3f}, {sample['gt_image'].max():.3f}]")
            print(f"  Filename: {sample['path']}")
            
            if i >= 2:
                break
        
        print("\n✓ RGB Adapter test passed!")
    else:
        print("No test samples found!")
