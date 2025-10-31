import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from typing import Optional

from model import CNN1D_fMRI
from data_loader import create_data_loader, load_data_for_laplacian, load_data_for_gradients
from utils import build_laplacian_from_data, compute_principal_gradient


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, use_amp=False, scaler=None):
    model.train()
    total_loss = total_lap_loss = n_batches = 0
    all_preds, all_labels, all_probs = [], [], []
    
    for batch_data in tqdm(dataloader, desc="Training"):
        x, y = batch_data[:2]
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp and scaler is not None:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(x)
                cls_loss = criterion(logits, y)
                lap_loss = model.compute_laplacian_regularization(lambda_scale=1.0)
                loss = cls_loss + lap_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(x)
            cls_loss = criterion(logits, y)
            lap_loss = model.compute_laplacian_regularization(lambda_scale=1.0)
            loss = cls_loss + lap_loss
            loss.backward()
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy() if probs.shape[1] == 2 else probs.detach().cpu().numpy())
        
        total_loss += loss.item()
        total_lap_loss += lap_loss.item()
        n_batches += 1
    
    metrics = {
        'loss': total_loss / n_batches,
        'lap_loss': total_lap_loss / n_batches,
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted')
    }
    
    if len(np.unique(all_labels)) == 2:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    
    return metrics


def validate(model, dataloader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = total_lap_loss = n_batches = 0
    all_preds, all_labels, all_probs, all_ids = [], [], [], []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 3:
                x, y, ids = batch_data
                all_ids.extend(ids.numpy() if isinstance(ids, torch.Tensor) else ids)
            else:
                x, y = batch_data
            
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            
            cls_loss = criterion(logits, y)
            lap_loss = model.compute_laplacian_regularization(lambda_scale=1.0)
            loss = cls_loss + lap_loss
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy() if probs.shape[1] == 2 else probs.cpu().numpy())
            
            total_loss += loss.item()
            total_lap_loss += lap_loss.item()
            n_batches += 1
    
    metrics = {
        'loss': total_loss / n_batches,
        'lap_loss': total_lap_loss / n_batches,
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    if len(np.unique(all_labels)) == 2:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    
    if return_predictions:
        metrics['predictions'] = all_preds
        metrics['labels'] = all_labels
        metrics['probabilities'] = all_probs
        if all_ids:
            metrics['ids'] = all_ids
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--source_tr', type=float, default=None)
    parser.add_argument('--target_tr', type=float, default=0.72)
    parser.add_argument('--n_rois', type=int, default=246)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--use_laplacian', action='store_true')
    parser.add_argument('--laplacian_lambda', type=float, default=0.01)
    parser.add_argument('--laplacian_top_k', type=int, default=10)
    parser.add_argument('--max_subjects_graph', type=int, default=200)
    parser.add_argument('--use_gradient_mixing', action='store_true')
    parser.add_argument('--temporal_pool', type=str, default='mean', choices=['mean', 'attention'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints_finetune')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--compile', action='store_true', help='Compile model with torch.compile')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_loader = create_data_loader(args.train_data, batch_size=args.batch_size, shuffle=True, 
                                     source_tr=args.source_tr, target_tr=args.target_tr if args.source_tr else None)
    val_loader = create_data_loader(args.val_data, batch_size=args.batch_size, shuffle=False, 
                                   source_tr=args.source_tr, target_tr=args.target_tr if args.source_tr else None) if args.val_data else None
    test_loader = create_data_loader(args.test_data, batch_size=args.batch_size, shuffle=False, 
                                    source_tr=args.source_tr, target_tr=args.target_tr if args.source_tr else None) if args.test_data else None
    
    model = CNN1D_fMRI(
        n_rois=args.n_rois,
        n_classes=args.n_classes,
        embedding_dim=args.embedding_dim,
        mode='finetune',
        use_laplacian=args.use_laplacian,
        use_gradient_mixing=args.use_gradient_mixing,
        laplacian_lambda=args.laplacian_lambda,
        laplacian_layer=2,
        temporal_pool=args.temporal_pool
    ).to(args.device)
    
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    if args.pretrained_model:
        pretrained_state = torch.load(args.pretrained_model, map_location=args.device)
        model_dict = model.state_dict()
        pretrained_dict = {f'encoder_layers.{i}.{p}': pretrained_state[f'encoder_layers.{i}.{p}'] 
                          for i in range(3) for p in ['weight', 'bias'] 
                          if f'encoder_layers.{i}.{p}' in pretrained_state}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    if args.use_laplacian:
        train_data = load_data_for_laplacian(args.train_data, max_subjects=args.max_subjects_graph)
        L = build_laplacian_from_data(train_data, top_k=args.laplacian_top_k, normalized=True, device=args.device)
        model.set_laplacian(L)
    
    if args.use_gradient_mixing:
        train_data = load_data_for_gradients(args.train_data, max_subjects=args.max_subjects_graph)
        gradient_coords, roi_order = compute_principal_gradient(train_data, n_components=3, max_subjects_for_fc=args.max_subjects_graph)
        model.set_roi_order(torch.from_numpy(roi_order).long().to(args.device))
        model.set_gradient_coords(torch.from_numpy(gradient_coords).float().to(args.device))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, fused=torch.cuda.is_available())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    scaler = GradScaler('cuda') if args.use_amp and args.device == 'cuda' else None
    
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device, 
                                   use_amp=args.use_amp, scaler=scaler)
        print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        if val_loader:
            val_metrics = validate(model, val_loader, criterion, args.device)
            print(f"Epoch {epoch}: Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_finetune_model.pt'))
        
        scheduler.step()
    
    if test_loader:
        best_model_path = os.path.join(args.save_dir, 'best_finetune_model.pt')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
        test_metrics = validate(model, test_loader, criterion, args.device, return_predictions=True)
        print(f"Test - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
        np.savez(os.path.join(args.save_dir, 'test_predictions.npz'), **{k: v for k, v in test_metrics.items() if k in ['predictions', 'labels', 'probabilities', 'ids']})
    
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_finetune_model.pt'))


if __name__ == '__main__':
    main()

