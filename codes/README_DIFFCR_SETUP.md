# DiffCR Testing on SEN12MS-CR Dataset (Kaggle)

This guide explains how to test the pretrained DiffCR model on the SEN12MS-CR dataset using Kaggle.

## Quick Start

### 1. Setup in Kaggle

```bash
# Clone your repo
cd /kaggle/working
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/DiffCR
```

### 2. Install Dependencies

```bash
pip install -q tifffile opencv-python tensorboard
```

### 3. Register the Dataset

Edit `data/__init__.py` and add:

```python
# At the top
from .sen12ms_adapter import SEN12MS_CR_RGB

# In define_dataloader function, add this case:
elif dataset_opt['which_dataset']['name'] == 'SEN12MS_CR_RGB':
    dataset = SEN12MS_CR_RGB(**dataset_opt['which_dataset']['args'])
```

### 4. Run Test

```bash
python run.py -p test -c config/sen12ms_cr_test.json -gpu 0
```

## Configuration

The config file `config/sen12ms_cr_test.json` is set up for:
- **Pretrained model**: `/kaggle/input/diffcr/pytorch/default/1/diffcr_new.pth`
- **Data CSV**: `/kaggle/working/data.csv`
- **Dataset**: SEN12MS-CR (RGB bands extracted automatically)

### Adjust Paths

Edit `config/sen12ms_cr_test.json` to match your Kaggle setup:
- `resume_state`: Path to your uploaded `diffcr_new.pth`
- `data_csv`: Path to your generated `data.csv`
- `data_root`: Path to your SEN12MS-CR dataset

## Testing on Subset

To test on only 50 samples (faster), edit `data/sen12ms_adapter.py`:

```python
# In SEN12MS_CR_RGB.__init__, uncomment this line:
self.filepair = self.filepair[:50]
```

## Expected Output

```
Begin model test.
Testing: 100%|████████████| N/N [XX:XX<00:00]

Results saved to: experiments/sen12ms_cr_diffcr_test/results/test/
```

## Performance

- **Full diffusion sampling**: ~2-5 seconds per image
- **50 samples**: ~5-10 minutes
- **All samples (~7000)**: ~4-10 hours

## Notes

- The adapter automatically extracts RGB bands [B4, B3, B2] from 13-band images
- Normalization is applied to match DiffCR's expected input range [-1, 1]
- Results are saved in the experiments directory
