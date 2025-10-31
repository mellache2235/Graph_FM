import torch
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional


def create_random_interval_mask(batch_size: int, n_timesteps: int, mask_ratio: float = 0.25,
                                min_interval_len: int = 5, max_interval_len: int = 20, device: str = 'cpu') -> torch.Tensor:
    masks = []
    for _ in range(batch_size):
        mask = torch.zeros(n_timesteps, dtype=torch.float32)
        masked_count = 0
        target_masked = int(n_timesteps * mask_ratio)
        attempts = 0
        while masked_count < target_masked and attempts < 100:
            interval_len = np.random.randint(min_interval_len, max_interval_len + 1)
            start = np.random.randint(0, n_timesteps - interval_len + 1)
            end = start + interval_len
            new_masked = (mask[start:end] == 0).sum().item()
            if masked_count + new_masked <= target_masked + interval_len // 2:
                mask[start:end] = 1
                masked_count += new_masked
            attempts += 1
        masks.append(mask)
    return torch.stack(masks).to(device)


def apply_mask_to_timeseries(x: torch.Tensor, mask: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    mask_expanded = mask.unsqueeze(-1)
    return x * (1 - mask_expanded) + mask_value * mask_expanded


def build_fc_adjacency_matrix(timeseries_data: np.ndarray, top_k: int = 10, threshold: Optional[float] = None) -> np.ndarray:
    n_subjects, n_timesteps, n_rois = timeseries_data.shape
    fc_matrices = []
    for subj in range(n_subjects):
        fc = np.corrcoef(timeseries_data[subj].T)
        np.fill_diagonal(fc, 0)
        fc_matrices.append(fc)
    fc_group = np.mean(fc_matrices, axis=0)
    
    if threshold is not None:
        fc_group[np.abs(fc_group) < threshold] = 0
    
    adj = np.zeros_like(fc_group)
    for i in range(n_rois):
        top_k_idx = np.argsort(np.abs(fc_group[i]))[-top_k:]
        adj[i, top_k_idx] = fc_group[i, top_k_idx]
    
    adj = (adj + adj.T) / 2
    return adj


def compute_graph_laplacian(adj: np.ndarray, normalized: bool = True) -> np.ndarray:
    degree = np.sum(adj, axis=1)
    if normalized:
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        L = np.eye(len(adj)) - D_inv_sqrt @ adj @ D_inv_sqrt
    else:
        D = np.diag(degree)
        L = D - adj
    return L


def compute_principal_gradient(timeseries_data: np.ndarray, n_components: int = 3, max_subjects_for_fc: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    # Gradient computation: subsample to max_subjects_for_fc to reduce computational cost
    # For 10,000+ windowed samples, using 200-500 subjects is sufficient for stable gradient estimates
    from sklearn.decomposition import PCA
    n_subjects, n_timesteps, n_rois = timeseries_data.shape
    
    # Subsample if needed
    if n_subjects > max_subjects_for_fc:
        print(f"Subsampling {max_subjects_for_fc} from {n_subjects} samples for gradient computation")
        indices = np.random.choice(n_subjects, max_subjects_for_fc, replace=False)
        timeseries_data = timeseries_data[indices]
        n_subjects = max_subjects_for_fc
    
    # Compute group-level FC (this is the expensive part)
    fc_matrices = []
    for subj in range(n_subjects):
        fc_matrices.append(np.corrcoef(timeseries_data[subj].T))
    fc_group = np.mean(fc_matrices, axis=0)
    
    affinity = cosine_similarity(fc_group)
    pca = PCA(n_components=n_components)
    gradient_coords = pca.fit_transform(affinity)
    roi_order = np.argsort(gradient_coords[:, 0])
    
    return gradient_coords, roi_order


def build_laplacian_from_data(timeseries_data: np.ndarray, top_k: int = 10, normalized: bool = True, device: str = 'cpu') -> torch.Tensor:
    adj = build_fc_adjacency_matrix(timeseries_data, top_k=top_k)
    L = compute_graph_laplacian(adj, normalized=normalized)
    return torch.from_numpy(L).float().to(device)


def resample_timeseries(timeseries: np.ndarray, source_tr: float, target_tr: float, kind: str = 'linear') -> np.ndarray:
    if source_tr == target_tr:
        return timeseries
    
    is_3d = timeseries.ndim == 3
    if not is_3d:
        timeseries = timeseries[np.newaxis, ...]
    
    n_subjects, n_timesteps, n_rois = timeseries.shape
    duration = (n_timesteps - 1) * source_tr
    new_n_timesteps = int(duration / target_tr) + 1
    
    source_times = np.arange(n_timesteps) * source_tr
    target_times = np.arange(new_n_timesteps) * target_tr
    
    resampled = np.zeros((n_subjects, new_n_timesteps, n_rois))
    for subj in range(n_subjects):
        for roi in range(n_rois):
            interpolator = interp1d(source_times, timeseries[subj, :, roi], kind=kind, bounds_error=False, fill_value='extrapolate')
            resampled[subj, :, roi] = interpolator(target_times)
    
    if not is_3d:
        resampled = resampled[0]
    
    return resampled


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str = 'mse') -> torch.Tensor:
    mask_expanded = mask.unsqueeze(-1)
    if loss_type == 'mse':
        element_loss = (pred - target) ** 2
    elif loss_type == 'mae':
        element_loss = torch.abs(pred - target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    masked_loss = element_loss * mask_expanded
    n_masked = mask_expanded.sum()
    if n_masked > 0:
        return masked_loss.sum() / n_masked
    return torch.tensor(0.0, device=pred.device)


