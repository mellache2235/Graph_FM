import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNN1D_fMRI
from data_loader import create_data_loader, load_data_for_laplacian, load_data_for_gradients
from utils import (
    create_random_interval_mask,
    apply_mask_to_timeseries,
    reconstruction_loss,
    build_laplacian_from_data,
    compute_principal_gradient
)


def pretrain_epoch(model, dataloader, optimizer, device, lambda_scale=1.0):
    model.train()
    total_recon_loss = total_lap_loss = total_loss = n_batches = 0
    
    for batch_data in tqdm(dataloader, desc="Pretraining"):
        x = batch_data.to(device)
        batch_size, n_timesteps, _ = x.shape
        
        mask = create_random_interval_mask(batch_size, n_timesteps, mask_ratio=0.25, min_interval_len=5, max_interval_len=20, device=device)
        x_masked = apply_mask_to_timeseries(x, mask, mask_value=0.0)
        reconstruction, _ = model(x_masked, mask=mask)
        
        recon_loss = reconstruction_loss(reconstruction, x, mask, loss_type='mse')
        lap_loss = model.compute_laplacian_regularization(lambda_scale=lambda_scale)
        loss = recon_loss + lap_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_recon_loss += recon_loss.item()
        total_lap_loss += lap_loss.item()
        total_loss += loss.item()
        n_batches += 1
    
    return {
        'recon_loss': total_recon_loss / n_batches,
        'lap_loss': total_lap_loss / n_batches,
        'total_loss': total_loss / n_batches
    }


def validate_pretrain(model, dataloader, device, lambda_scale=1.0):
    model.eval()
    total_recon_loss = total_lap_loss = total_loss = n_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            x = batch_data.to(device)
            batch_size, n_timesteps, _ = x.shape
            
            mask = create_random_interval_mask(batch_size, n_timesteps, mask_ratio=0.25, min_interval_len=5, max_interval_len=20, device=device)
            x_masked = apply_mask_to_timeseries(x, mask, mask_value=0.0)
            reconstruction, _ = model(x_masked, mask=mask)
            
            recon_loss = reconstruction_loss(reconstruction, x, mask, loss_type='mse')
            lap_loss = model.compute_laplacian_regularization(lambda_scale=lambda_scale)
            loss = recon_loss + lap_loss
            
            total_recon_loss += recon_loss.item()
            total_lap_loss += lap_loss.item()
            total_loss += loss.item()
            n_batches += 1
    
    return {
        'recon_loss': total_recon_loss / n_batches,
        'lap_loss': total_lap_loss / n_batches,
        'total_loss': total_loss / n_batches
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--n_rois', type=int, default=246)
    parser.add_argument('--n_timesteps', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--use_laplacian', action='store_true')
    parser.add_argument('--laplacian_lambda', type=float, default=0.01)
    parser.add_argument('--laplacian_top_k', type=int, default=10)
    parser.add_argument('--laplacian_warmup_epochs', type=int, default=10)
    parser.add_argument('--max_subjects_graph', type=int, default=200, help='Max subjects for graph computation')
    parser.add_argument('--use_gradient_mixing', action='store_true')
    parser.add_argument('--temporal_pool', type=str, default='mean', choices=['mean', 'attention'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_loader = create_data_loader(args.train_data, batch_size=args.batch_size, shuffle=True, pretraining=True)
    val_loader = create_data_loader(args.val_data, batch_size=args.batch_size, shuffle=False, pretraining=True) if args.val_data else None
    
    model = CNN1D_fMRI(
        n_rois=args.n_rois,
        n_timesteps=args.n_timesteps,
        embedding_dim=args.embedding_dim,
        mode='pretrain',
        use_laplacian=args.use_laplacian,
        use_gradient_mixing=args.use_gradient_mixing,
        laplacian_lambda=args.laplacian_lambda,
        laplacian_layer=2,
        temporal_pool=args.temporal_pool
    ).to(args.device)
    
    if args.use_laplacian:
        train_data = load_data_for_laplacian(args.train_data, max_subjects=args.max_subjects_graph)
        L = build_laplacian_from_data(train_data, top_k=args.laplacian_top_k, normalized=True, device=args.device)
        model.set_laplacian(L)
    
    if args.use_gradient_mixing:
        train_data = load_data_for_gradients(args.train_data, max_subjects=args.max_subjects_graph)
        gradient_coords, roi_order = compute_principal_gradient(train_data, n_components=3, max_subjects_for_fc=args.max_subjects_graph)
        model.set_roi_order(torch.from_numpy(roi_order).long().to(args.device))
        model.set_gradient_coords(torch.from_numpy(gradient_coords).float().to(args.device))
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        # Lambda warmup: 0 → 1 over warmup_epochs to prevent over-smoothing early on
        if args.use_laplacian and args.laplacian_warmup_epochs > 0:
            lambda_scale = min(1.0, epoch / args.laplacian_warmup_epochs)
        else:
            lambda_scale = 1.0
        
        train_metrics = pretrain_epoch(model, train_loader, optimizer, args.device, lambda_scale=lambda_scale)
        print(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f} (λ_scale: {lambda_scale:.3f})")
        
        if val_loader:
            val_metrics = validate_pretrain(model, val_loader, args.device, lambda_scale=lambda_scale)
            print(f"Epoch {epoch}: Val Loss: {val_metrics['total_loss']:.4f}")
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_pretrain_model.pt'))
        
        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_pretrain_model.pt'))


if __name__ == '__main__':
    main()

