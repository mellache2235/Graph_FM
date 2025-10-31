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
        
        data = load_pklz(data_path)
        if isinstance(data, dict):
            self.X = data['X']
            self.Y = data['Y']
            self.ids = data.get('ids', None)
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
        
        if isinstance(self.X, list):
            self.X = np.array(self.X)
        if isinstance(self.Y, list):
            self.Y = np.array(self.Y)
        if self.ids is not None and isinstance(self.ids, list):
            self.ids = np.array(self.ids)
        
        if self.X.ndim == 3 and self.X.shape[1] < self.X.shape[2]:
            self.X = self.X.transpose(0, 2, 1)
        
        if source_tr is not None and target_tr is not None:
            self.X = resample_timeseries(self.X, source_tr=source_tr, target_tr=target_tr, kind=interp_kind)
        
        assert len(self.X) == len(self.Y), "X and Y must have same length"
        if self.ids is not None:
            assert len(self.X) == len(self.ids), "X and ids must have same length"
    
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
        data = load_pklz(data_path)
        
        if isinstance(data, dict):
            self.X = data['X']
        elif isinstance(data, (tuple, list)):
            self.X = data[0]
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        if isinstance(self.X, list):
            self.X = np.array(self.X)
        
        if self.X.ndim == 3 and self.X.shape[1] < self.X.shape[2]:
            self.X = self.X.transpose(0, 2, 1)
    
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
    data = load_pklz(data_path)
    if isinstance(data, dict):
        X = data['X']
    elif isinstance(data, (tuple, list)):
        X = data[0]
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    
    if isinstance(X, list):
        X = np.array(X)
    
    if X.ndim == 3 and X.shape[1] < X.shape[2]:
        X = X.transpose(0, 2, 1)
    
    if max_subjects is not None and len(X) > max_subjects:
        indices = np.random.choice(len(X), max_subjects, replace=False)
        X = X[indices]
    
    return X


def load_data_for_gradients(data_path: str, max_subjects: Optional[int] = None) -> np.ndarray:
    return load_data_for_laplacian(data_path, max_subjects)


