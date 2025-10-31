import pickle
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any
from utils import resample_timeseries


def load_pklz(file_path: str) -> Any:
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_npz(file_path: str) -> Dict:
    """Load .npz file and return as dictionary."""
    return dict(np.load(file_path, allow_pickle=True))


def load_data_file(file_path: str) -> Any:
    """Load data from either .pklz or .npz file."""
    if file_path.endswith('.npz'):
        return load_npz(file_path)
    elif file_path.endswith('.pklz') or file_path.endswith('.pkl.gz'):
        return load_pklz(file_path)
    else:
        # Try to detect by attempting to load
        try:
            return load_npz(file_path)
        except:
            return load_pklz(file_path)


def save_pklz(data: Any, file_path: str):
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)


class fMRIDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        load_ids: bool = False,
        source_tr: Optional[float] = None,
        target_tr: Optional[float] = None,
        interp_kind: str = 'linear'
    ):
        self.transform = transform
        self.load_ids = load_ids
        self.source_tr = source_tr
        self.target_tr = target_tr
        self.interp_kind = interp_kind
        
        data = load_data_file(data_path)
        
        # Handle different key formats
        if isinstance(data, dict):
            # Try different key variations for data
            self.X = None
            for key in ['data', 'Data', 'DATA', 'X', 'x', 'data_train_window', 'data_test_window']:
                if key in data:
                    self.X = data[key]
                    break
            if self.X is None:
                raise ValueError(f"Could not find data key. Available keys: {list(data.keys())}")
            
            # Try different key variations for labels
            self.Y = None
            for key in ['label', 'Label', 'LABEL', 'labels', 'Labels', 'Y', 'y', 'labels_train_window', 'labels_test_window']:
                if key in data:
                    self.Y = data[key]
                    break
            if self.Y is None:
                raise ValueError(f"Could not find label key. Available keys: {list(data.keys())}")
            
            # Try to get IDs
            self.ids = None
            for key in ['ids', 'IDs', 'id', 'ID', 'subject_ids', 'subjects']:
                if key in data:
                    self.ids = data[key]
                    break
            
            # Try to get TR from data if not provided
            if source_tr is None and 'tr' in data:
                source_tr = float(data['tr'])
                self.source_tr = source_tr
                print(f"Using TR from data file: {source_tr}s")
                    
        elif isinstance(data, (tuple, list)):
            if len(data) == 2:
                self.X, self.Y = data
                self.ids = None
            elif len(data) == 3:
                self.X, self.Y, self.ids = data
            else:
                raise ValueError(f"Expected tuple of length 2 or 3, got {len(data)}")
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        # Convert to numpy arrays
        if isinstance(self.X, list):
            self.X = np.array(self.X)
        if isinstance(self.Y, list):
            self.Y = np.array(self.Y)
        if self.ids is not None and isinstance(self.ids, list):
            self.ids = np.array(self.ids)
        
        # Handle variable-length timeseries (pad or truncate to same length)
        if self.X.ndim == 2:
            # Already 2D, assume (n_samples, features)
            pass
        elif self.X.ndim == 3:
            # Check if shapes are consistent
            if self.X.shape[1] < self.X.shape[2]:
                # (n_subjects, n_rois, n_timesteps) -> (n_subjects, n_timesteps, n_rois)
                self.X = self.X.transpose(0, 2, 1)
        elif self.X.dtype == object:
            # Variable length timeseries stored as object array
            self.X = self._equalize_lengths(self.X)
        
        # Remove NaNs
        self.X, self.Y, self.ids = self._remove_nans(self.X, self.Y, self.ids)
        
        # Resample TR if needed
        if source_tr is not None and target_tr is not None:
            self.X = resample_timeseries(self.X, source_tr=source_tr, target_tr=target_tr, kind=interp_kind)
        
        assert len(self.X) == len(self.Y), "X and Y must have same length"
        if self.ids is not None:
            assert len(self.X) == len(self.ids), "X and ids must have same length"
        
        print(f"Loaded {len(self.X)} samples, X shape: {self.X.shape}, Y shape: {self.Y.shape}")
    
    def _equalize_lengths(self, X_obj):
        """Handle variable-length timeseries by padding/truncating."""
        if not isinstance(X_obj[0], np.ndarray):
            return np.array(X_obj)
        
        # Find max length
        lengths = [x.shape[0] for x in X_obj]
        max_len = max(lengths)
        n_rois = X_obj[0].shape[1] if X_obj[0].ndim > 1 else 1
        
        # Pad/truncate to max length
        X_uniform = []
        for x in X_obj:
            if x.shape[0] < max_len:
                # Pad with zeros or repeat last timepoint
                pad_width = [(0, max_len - x.shape[0])]
                if x.ndim > 1:
                    pad_width.append((0, 0))
                x_padded = np.pad(x, pad_width, mode='edge')
                X_uniform.append(x_padded)
            elif x.shape[0] > max_len:
                # Truncate
                X_uniform.append(x[:max_len])
            else:
                X_uniform.append(x)
        
        return np.array(X_uniform)
    
    def _remove_nans(self, X, Y, ids):
        """Remove samples with NaN values in X or Y."""
        # Find samples with NaNs in X
        if X.ndim == 3:
            # Check across all timesteps and ROIs
            nan_mask_x = np.any(np.isnan(X).reshape(len(X), -1), axis=1)
        else:
            nan_mask_x = np.any(np.isnan(X), axis=1) if X.ndim > 1 else np.isnan(X)
        
        # Find samples with NaNs in Y
        nan_mask_y = np.isnan(Y) if Y.ndim == 1 else np.any(np.isnan(Y), axis=1)
        
        # Combine masks
        valid_mask = ~(nan_mask_x | nan_mask_y)
        n_removed = (~valid_mask).sum()
        
        if n_removed > 0:
            print(f"Removing {n_removed} samples with NaN values")
        
        # Filter data
        X_clean = X[valid_mask]
        Y_clean = Y[valid_mask]
        ids_clean = ids[valid_mask] if ids is not None else None
        
        return X_clean, Y_clean, ids_clean
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple:
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(np.array(self.Y[idx]))
        
        if y.dtype == torch.float64:
            y = y.float()
        elif y.ndim == 0:
            y = y.long()
        
        if self.transform is not None:
            x = self.transform(x)
        
        if self.load_ids and self.ids is not None:
            return x, y, self.ids[idx]
        return x, y
    
    def get_data_info(self) -> Dict[str, Any]:
        return {
            'n_subjects': len(self.X),
            'n_timesteps': self.X.shape[1],
            'n_rois': self.X.shape[2],
            'n_classes': len(np.unique(self.Y)) if self.Y.ndim == 1 else self.Y.shape[1],
            'has_ids': self.ids is not None
        }


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[callable] = None):
        self.transform = transform
        data = load_data_file(data_path)
        
        if isinstance(data, dict):
            # Try different key variations (prioritize pretrain-specific keys)
            self.X = None
            for key in ['data_train_window', 'data', 'Data', 'DATA', 'X', 'x']:
                if key in data:
                    self.X = data[key]
                    break
            if self.X is None:
                raise ValueError(f"Could not find data key. Available keys: {list(data.keys())}")
        elif isinstance(data, (tuple, list)):
            self.X = data[0]
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        if isinstance(self.X, list):
            self.X = np.array(self.X)
        
        # Handle variable-length timeseries
        if self.X.dtype == object:
            self.X = self._equalize_lengths(self.X)
        
        if self.X.ndim == 3 and self.X.shape[1] < self.X.shape[2]:
            self.X = self.X.transpose(0, 2, 1)
        
        # Remove NaNs
        self.X = self._remove_nans(self.X)
        
        print(f"Loaded {len(self.X)} samples for pretraining, X shape: {self.X.shape}")
    
    def _equalize_lengths(self, X_obj):
        """Handle variable-length timeseries."""
        if not isinstance(X_obj[0], np.ndarray):
            return np.array(X_obj)
        
        lengths = [x.shape[0] for x in X_obj]
        max_len = max(lengths)
        
        X_uniform = []
        for x in X_obj:
            if x.shape[0] < max_len:
                pad_width = [(0, max_len - x.shape[0])]
                if x.ndim > 1:
                    pad_width.append((0, 0))
                x_padded = np.pad(x, pad_width, mode='edge')
                X_uniform.append(x_padded)
            elif x.shape[0] > max_len:
                X_uniform.append(x[:max_len])
            else:
                X_uniform.append(x)
        
        return np.array(X_uniform)
    
    def _remove_nans(self, X):
        """Remove samples with NaN values."""
        if X.ndim == 3:
            nan_mask = np.any(np.isnan(X).reshape(len(X), -1), axis=1)
        else:
            nan_mask = np.any(np.isnan(X), axis=1) if X.ndim > 1 else np.isnan(X)
        
        valid_mask = ~nan_mask
        n_removed = (~valid_mask).sum()
        
        if n_removed > 0:
            print(f"Removing {n_removed} samples with NaN values")
        
        return X[valid_mask]
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        x = torch.from_numpy(self.X[idx]).float()
        if self.transform is not None:
            x = self.transform(x)
        return x
    
    def get_data_info(self) -> Dict[str, Any]:
        return {
            'n_subjects': len(self.X),
            'n_timesteps': self.X.shape[1],
            'n_rois': self.X.shape[2]
        }


def create_data_loader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    load_ids: bool = False,
    transform: Optional[callable] = None,
    pretraining: bool = False,
    source_tr: Optional[float] = None,
    target_tr: Optional[float] = None,
    interp_kind: str = 'linear'
) -> DataLoader:
    if pretraining:
        dataset = PretrainDataset(data_path, transform=transform)
    else:
        dataset = fMRIDataset(data_path, transform=transform, load_ids=load_ids,
                            source_tr=source_tr, target_tr=target_tr, interp_kind=interp_kind)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def load_data_for_laplacian(data_path: str, max_subjects: Optional[int] = None) -> np.ndarray:
    data = load_data_file(data_path)
    if isinstance(data, dict):
        X = None
        for key in ['data', 'Data', 'DATA', 'X', 'x', 'data_train_window', 'data_test_window']:
            if key in data:
                X = data[key]
                break
        if X is None:
            raise ValueError(f"Could not find data key. Available keys: {list(data.keys())}")
    elif isinstance(data, (tuple, list)):
        X = data[0]
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    
    if isinstance(X, list):
        X = np.array(X)
    
    # Handle variable-length timeseries
    if X.dtype == object:
        X = _equalize_lengths_helper(X)
    
    if X.ndim == 3 and X.shape[1] < X.shape[2]:
        X = X.transpose(0, 2, 1)
    
    # Remove NaNs
    if X.ndim == 3:
        nan_mask = np.any(np.isnan(X).reshape(len(X), -1), axis=1)
    else:
        nan_mask = np.any(np.isnan(X), axis=1) if X.ndim > 1 else np.isnan(X)
    X = X[~nan_mask]
    
    if max_subjects is not None and len(X) > max_subjects:
        indices = np.random.choice(len(X), max_subjects, replace=False)
        X = X[indices]
    
    return X


def _equalize_lengths_helper(X_obj):
    """Helper function to equalize variable-length timeseries."""
    if not isinstance(X_obj[0], np.ndarray):
        return np.array(X_obj)
    
    lengths = [x.shape[0] for x in X_obj]
    max_len = max(lengths)
    
    X_uniform = []
    for x in X_obj:
        if x.shape[0] < max_len:
            pad_width = [(0, max_len - x.shape[0])]
            if x.ndim > 1:
                pad_width.append((0, 0))
            x_padded = np.pad(x, pad_width, mode='edge')
            X_uniform.append(x_padded)
        elif x.shape[0] > max_len:
            X_uniform.append(x[:max_len])
        else:
            X_uniform.append(x)
    
    return np.array(X_uniform)


def load_data_for_gradients(data_path: str, max_subjects: Optional[int] = None) -> np.ndarray:
    return load_data_for_laplacian(data_path, max_subjects)


