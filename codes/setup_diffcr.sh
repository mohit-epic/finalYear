#!/bin/bash
# DiffCR Setup Script for Kaggle
# Run this after cloning the repo to set up DiffCR properly

echo "Setting up DiffCR for testing..."

# Navigate to DiffCR directory
cd /kaggle/working/finalYear/codes/DiffCR || exit 1

# Copy fixed network file
echo "Copying fixed network_x0_dpm_solver.py..."
cp /kaggle/working/finalYear/codes/DiffCR_models/network_x0_dpm_solver.py models/network_x0_dpm_solver.py

# Copy modified model.py with metric display
echo "Copying modified model.py with metric display..."
cp /kaggle/working/finalYear/codes/DiffCR_models/model.py models/model.py

# Copy adapter
echo "Copying SEN12MS-CR adapter..."
cp /kaggle/working/finalYear/codes/diffcr_sen12ms_adapter.py data/sen12ms_adapter.py

# Copy config
echo "Copying test config..."
cp /kaggle/working/finalYear/codes/diffcr_sen12ms_cr_test.json config/sen12ms_cr_test.json

# Update data/__init__.py to register dataset
echo "Registering SEN12MS_CR_RGB dataset..."
if ! grep -q "from .sen12ms_adapter import SEN12MS_CR_RGB" data/__init__.py; then
    sed -i '2a from .sen12ms_adapter import SEN12MS_CR_RGB' data/__init__.py
fi

# Add dataset case in define_dataset function
if ! grep -q "SEN12MS_CR_RGB" data/__init__.py; then
    # Find the line with "def define_dataset" and add the elif case
    sed -i '/val_dataset_opt = opt/a\    elif dataset_opt["which_dataset"]["name"] == "SEN12MS_CR_RGB":\n        dataset = SEN12MS_CR_RGB(**dataset_opt["which_dataset"]["args"])' data/__init__.py
fi

echo "✅ DiffCR setup complete!"
echo ""
echo "Now run:"
echo "  pip install -q tifffile"
echo "  python run.py -p test -c config/sen12ms_cr_test.json -gpu 0"

