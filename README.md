# Graph-Informed 1D CNN for fMRI Timeseries

A PyTorch implementation of 1D CNN for fMRI timeseries analysis with graph-informed regularization techniques. Supports BERT-style pretraining with masked reconstruction and supervised finetuning for classification.

## Features

### Model Architecture
- **Pretraining**: 6-layer 1D CNN (246→512→512→512→246→246→246) with masked interval reconstruction
- **Finetuning**: 3-layer 1D CNN with temporal averaging and classification head
- Pretrained embeddings extracted from layer 3 (512-dimensional)

### Graph-Informed Regularization

#### 1. Laplacian Smoothing Regularizer
- Builds group-level functional connectivity (FC) adjacency matrix from training data
- Computes graph Laplacian: L = I - D^(-1/2) A D^(-1/2)
- Adds regularization term: λ * tr(H^T L H) on early hidden representations
- Encourages smooth activations across functionally connected ROIs

#### 2. Gradient-Aware ROI Mixing
Two complementary approaches:
- **Spatial ordering**: Sorts ROIs by principal gradient and applies small 2D convolution
- **ROI gating**: Uses MLP to gate ROIs based on gradient coordinates
- Captures hierarchical organization of brain networks

### Data Format
- Input: `.pklz` (gzipped pickle) files
- Supports:
  - Dictionary: `{'X': data, 'Y': labels, 'ids': subject_ids}`
  - Tuple: `(X, Y)` or `(X, Y, ids)`
- Automatic reshaping from `(n_subjects, n_channels, n_timesteps)` to `(n_subjects, n_timesteps, n_channels)`

## Installation

```bash
# Clone repository
git clone https://github.com/mellache2235/Graph_FM.git
cd Graph_FM

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

The implementation supports both PyTorch and PyTorch Lightning. **PyTorch Lightning is recommended** for GPU acceleration, automatic mixed precision, and cleaner code.

## Usage

### 1. Pretraining

```bash
python pretrain.py \
    --train_data data/pretrain_train.pklz \
    --val_data data/pretrain_val.pklz \
    --n_rois 246 \
    --n_timesteps 200 \
    --embedding_dim 512 \
    --use_laplacian \
    --laplacian_lambda 0.01 \
    --laplacian_top_k 10 \
    --use_gradient_mixing \
    --mask_ratio 0.25 \
    --min_interval 5 \
    --max_interval 20 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-3 \
    --save_dir checkpoints/pretrain
```

**Key Arguments:**
- `--mask_ratio`: Proportion of timesteps to mask (default: 0.25)
- `--min_interval`, `--max_interval`: Range for masked interval lengths
- `--use_laplacian`: Enable Laplacian smoothing regularization
- `--laplacian_lambda`: Weight for Laplacian regularization
- `--laplacian_top_k`: Number of top connections per ROI in adjacency matrix
- `--use_gradient_mixing`: Enable gradient-aware ROI mixing

### 2. Finetuning

```bash
python finetune.py \
    --train_data data/finetune_train.pklz \
    --val_data data/finetune_val.pklz \
    --test_data data/finetune_test.pklz \
    --n_rois 246 \
    --n_classes 2 \
    --embedding_dim 512 \
    --pretrained_model checkpoints/pretrain/best_pretrain_model.pt \
    --freeze_encoder \
    --use_laplacian \
    --laplacian_lambda 0.01 \
    --use_gradient_mixing \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --dropout 0.5 \
    --track_ids \
    --save_dir checkpoints/finetune
```

**Key Arguments:**
- `--pretrained_model`: Path to pretrained model weights
- `--freeze_encoder`: Freeze encoder layers (transfer learning)
- `--track_ids`: Track subject IDs for cross-validation/analysis
- `--dropout`: Dropout rate before classifier

### 3. Testing on External Datasets

Test a trained model on new data:

```bash
python test.py \
    --model_path checkpoints/finetune/best_finetune_model.ckpt \
    --test_data data/external_test.pklz \
    --n_rois 246 \
    --n_classes 2 \
    --embedding_dim 512 \
    --source_tr 2.0 \
    --target_tr 0.72 \
    --device auto \
    --output_dir test_results
```

**Key Features:**
- Automatic GPU acceleration
- TR resampling for datasets with different acquisition parameters
- Comprehensive metrics: accuracy, macro-precision, macro-recall, macro-F1
- Saves predictions and confusion matrix

### 4. Repeated Evaluation (50x Train-Valid-Test Splits)

For robust model evaluation with statistical significance:

```bash
python repeated_evaluation.py \
    --data_path data/dataset.pklz \
    --n_repeats 50 \
    --source_tr 2.0 \
    --target_tr 0.72 \
    --n_rois 246 \
    --n_classes 2 \
    --pretrained_model checkpoints/pretrain/best_pretrain_model.pt \
    --use_laplacian \
    --use_gradient_mixing \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --save_dir repeated_eval_results
```

**Evaluation Protocol:**
- 50 repeated train-valid-test splits (64%-16%-20%)
- No early stopping - trains for full epoch count
- Metrics tracked: accuracy, macro-precision, macro-recall, macro-F1
- Results saved as CSV with mean ± std
- Uses PyTorch Lightning for GPU acceleration

### 5. Python API

```python
import torch
from model import CNN1D_fMRI
from data_loader import create_data_loader
from utils import build_laplacian_from_data, compute_principal_gradient

# Load data
train_loader = create_data_loader(
    'data/train.pklz',
    batch_size=32,
    shuffle=True,
    pretraining=False  # Set True for pretraining
)

# Create model
model = CNN1D_fMRI(
    n_rois=246,
    n_classes=2,
    n_timesteps=200,
    embedding_dim=512,
    mode='finetune',  # or 'pretrain'
    use_laplacian=True,
    use_gradient_mixing=True,
    laplacian_lambda=0.01
)

# Setup Laplacian regularization
from data_loader import load_data_for_laplacian
train_data = load_data_for_laplacian('data/train.pklz', max_subjects=500)
L = build_laplacian_from_data(train_data, top_k=10)
model.set_laplacian(L)

# Setup gradient-aware mixing
from data_loader import load_data_for_gradients
train_data = load_data_for_gradients('data/train.pklz', max_subjects=500)
gradient_coords, roi_order = compute_principal_gradient(train_data)
model.set_roi_order(torch.from_numpy(roi_order).long())
model.set_gradient_coords(torch.from_numpy(gradient_coords).float())

# Training
for x, y in train_loader:
    logits, _ = model(x)
    loss = criterion(logits, y) + model.compute_laplacian_regularization()
    # ... backward pass
```

### 6. Using PyTorch Lightning (Recommended)

```python
import pytorch_lightning as pl
from model_lightning import LitfMRIModel
from data_loader import create_data_loader

# Create model
model = LitfMRIModel(
    n_rois=246,
    n_classes=2,
    embedding_dim=512,
    mode='finetune',
    use_laplacian=True,
    use_gradient_mixing=True,
    lr=1e-4,
    scheduler='cosine'
)

# Setup Laplacian and gradients
from utils import build_laplacian_from_data, compute_principal_gradient
from data_loader import load_data_for_laplacian

train_data = load_data_for_laplacian('data/train.pklz')
L = build_laplacian_from_data(train_data, top_k=10)
model.set_laplacian(L)

# Create data loaders
train_loader = create_data_loader('data/train.pklz', batch_size=32)
val_loader = create_data_loader('data/val.pklz', batch_size=32, shuffle=False)

# Setup trainer with GPU
trainer = pl.Trainer(
    max_epochs=50,
    accelerator='auto',  # Automatically uses GPU if available
    devices=1,
    precision='16-mixed'  # Automatic mixed precision for faster training
)

# Train
trainer.fit(model, train_loader, val_loader)

# Test
test_loader = create_data_loader('data/test.pklz', batch_size=32, shuffle=False)
trainer.test(model, test_loader)
```

## File Structure

```
Graph_FM/
├── model.py                  # Base CNN1D_fMRI model
├── model_lightning.py        # PyTorch Lightning wrapper (recommended)
├── data_loader.py            # Data loading utilities for .pklz files
├── utils.py                  # Graph utilities and TR resampling
├── pretrain.py               # Pretraining script
├── finetune.py               # Finetuning script
├── test.py                   # Test on external datasets
├── repeated_evaluation.py    # 50x repeated evaluation protocol
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── example_usage.py          # Usage examples
└── examples/                 # Additional examples
```

## TR Resampling

The implementation handles different TRs (repetition times) automatically:

- **Pretraining**: Typically at TR=0.72s
- **Finetuning**: May use different datasets (e.g., ADHD at TR=2.0s)

Data is automatically resampled using scipy.interpolate.interp1d:

```python
# Automatic resampling when loading data
train_loader = create_data_loader(
    'adhd_data.pklz',
    source_tr=2.0,      # ADHD data at 2.0s TR
    target_tr=0.72,     # Resample to match pretrained model
    interp_kind='linear'
)

# Manual resampling
from utils import resample_timeseries
X_resampled = resample_timeseries(
    X,                  # (subjects, timesteps, rois)
    source_tr=2.0,
    target_tr=0.72,
    kind='linear'       # 'linear', 'cubic', or 'nearest'
)
```

## GPU Support

All scripts automatically use GPU when available:

```bash
# PyTorch Lightning automatically detects and uses GPU
python repeated_evaluation.py ...  # Uses GPU if available

# For test script, specify device
python test.py ... --device auto  # or 'gpu' or 'cpu'
```

## Key Components

### Laplacian Smoothing

1. **Build adjacency matrix** from functional connectivity:
   - Compute correlation matrices across subjects
   - Average to get group-level FC
   - Keep top-k connections per ROI

2. **Compute Laplacian**: L = I - D^(-1/2) A D^(-1/2)

3. **Add regularization**: λ * tr(H^T L H)
   - Applied to hidden representations at specified layer
   - Encourages smooth activations across connected ROIs

### Gradient-Aware Mixing

1. **Compute principal gradients**:
   - Build affinity matrix from FC patterns
   - Apply PCA to get gradient components
   - Sort ROIs by first principal component

2. **Apply mixing**:
   - 2D convolution across ordered ROI dimension
   - MLP gating based on gradient coordinates

## Data Preparation

Your `.pklz` files should contain:

```python
import gzip
import pickle
import numpy as np

# Format 1: Dictionary
data = {
    'X': np.random.randn(100, 200, 246),  # (subjects, timesteps, rois)
    'Y': np.random.randint(0, 2, 100),     # (subjects,)
    'ids': np.arange(100)                  # (subjects,) - optional
}

# Format 2: Tuple
data = (X, Y) or (X, Y, ids)

# Save
with gzip.open('data.pklz', 'wb') as f:
    pickle.dump(data, f)
```

**Note**: If your data is in `(subjects, channels, timesteps)` format, it will be automatically reshaped to `(subjects, timesteps, channels)`.

## Learning Rate Guidelines

The implementation enforces that maximum LR doesn't exceed 1e-3 (the pretraining rate):

```bash
# Constant LR (baseline)
--lr 1e-4 --scheduler constant

# Cosine annealing with warmup (recommended)
--lr 1e-4 --max_lr 1e-3 --min_lr 1e-6 --scheduler cosine --warmup_epochs 5

# Reduce on plateau (adaptive)
--lr 1e-4 --scheduler plateau

# OneCycle (fast convergence - experimental)
--lr 1e-5 --max_lr 8e-4 --scheduler onecycle
```

**Note**: All encoder layers are finetuned by default for best performance. Use `--freeze_encoder` only for very small datasets.

## Performance Tips

1. **Use PyTorch Lightning**: Automatic GPU acceleration, mixed precision, and better performance
2. **TR Resampling**: Linear interpolation is fast and effective; cubic may be better for smooth data
3. **Laplacian Computation**: Use `max_subjects=500` to limit memory usage for large datasets
4. **Gradient Mixing**: Compute gradients once per dataset and reuse across splits
5. **Data Loading**: Use `--num_workers 4` or more for faster data loading
6. **Mixed Precision**: PyTorch Lightning supports `precision='16-mixed'` for 2x speedup
7. **Batch Size**: Use largest batch size that fits in GPU memory (typically 32-64)

## Citation

If you use this code, please cite:

```bibtex
@software{graph_fm_2025,
  title={Graph-Informed 1D CNN for fMRI Timeseries},
  author={Your Name},
  year={2025},
  url={https://github.com/mellache2235/Graph_FM}
}
```

## License

MIT License

## Acknowledgments

- Laplacian smoothing inspired by graph neural network literature
- Gradient-aware mixing based on principal gradient analysis
- BERT-style pretraining adapted for fMRI timeseries

