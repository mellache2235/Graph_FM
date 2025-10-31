"""
PyTorch Lightning wrapper for 1D CNN fMRI model
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any

from model import CNN1D_fMRI
from utils import create_random_interval_mask, apply_mask_to_timeseries, reconstruction_loss


class LitfMRIModel(pl.LightningModule):
    """PyTorch Lightning module for fMRI 1D CNN."""
    
    def __init__(
        self,
        n_rois: int = 246,
        n_classes: int = 2,
        n_timesteps: int = 200,
        embedding_dim: int = 512,
        kernel_size: int = 3,
        mode: str = 'finetune',
        use_laplacian: bool = False,
        use_gradient_mixing: bool = False,
        laplacian_lambda: float = 0.01,
        laplacian_layer: int = 2,
        dropout: float = 0.5,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_lr: float = 1e-3,
        min_lr: float = 1e-6,
        scheduler: str = 'cosine',
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        # Pretraining specific
        mask_ratio: float = 0.25,
        min_interval: int = 5,
        max_interval: int = 20
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = CNN1D_fMRI(
            n_rois=n_rois,
            n_classes=n_classes,
            n_timesteps=n_timesteps,
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            mode=mode,
            use_laplacian=use_laplacian,
            use_gradient_mixing=use_gradient_mixing,
            laplacian_lambda=laplacian_lambda,
            laplacian_layer=laplacian_layer,
            dropout=dropout
        )
        
        # Loss function
        if mode == 'finetune':
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x, mask=None, return_embedding=False):
        return self.model(x, mask=mask, return_embedding=return_embedding)
    
    def set_laplacian(self, L):
        """Set Laplacian matrix."""
        self.model.set_laplacian(L)
    
    def set_roi_order(self, order):
        """Set ROI ordering."""
        self.model.set_roi_order(order)
    
    def set_gradient_coords(self, coords):
        """Set gradient coordinates."""
        self.model.set_gradient_coords(coords)
    
    def training_step(self, batch, batch_idx):
        if self.hparams.mode == 'pretrain':
            x = batch
            batch_size, n_timesteps, n_rois = x.shape
            
            # Create mask
            mask = create_random_interval_mask(
                batch_size, n_timesteps,
                mask_ratio=self.hparams.mask_ratio,
                min_interval_len=self.hparams.min_interval,
                max_interval_len=self.hparams.max_interval,
                device=self.device
            )
            
            # Masked input
            x_masked = apply_mask_to_timeseries(x, mask, mask_value=0.0)
            
            # Forward
            reconstruction, _ = self(x_masked, mask=mask)
            
            # Losses
            recon_loss = reconstruction_loss(reconstruction, x, mask, loss_type='mse')
            lap_loss = self.model.compute_laplacian_regularization()
            loss = recon_loss + lap_loss
            
            self.log('train/recon_loss', recon_loss, prog_bar=True)
            self.log('train/lap_loss', lap_loss)
            self.log('train/loss', loss, prog_bar=True)
            
        else:  # finetune
            x, y = batch[:2]  # Ignore IDs if present
            
            # Forward
            logits, _ = self(x)
            
            # Losses
            cls_loss = self.criterion(logits, y)
            lap_loss = self.model.compute_laplacian_regularization()
            loss = cls_loss + lap_loss
            
            # Metrics
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            
            self.log('train/cls_loss', cls_loss, prog_bar=True)
            self.log('train/lap_loss', lap_loss)
            self.log('train/loss', loss, prog_bar=True)
            self.log('train/acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.hparams.mode == 'pretrain':
            x = batch
            batch_size, n_timesteps, n_rois = x.shape
            
            mask = create_random_interval_mask(
                batch_size, n_timesteps,
                mask_ratio=self.hparams.mask_ratio,
                min_interval_len=self.hparams.min_interval,
                max_interval_len=self.hparams.max_interval,
                device=self.device
            )
            
            x_masked = apply_mask_to_timeseries(x, mask, mask_value=0.0)
            reconstruction, _ = self(x_masked, mask=mask)
            
            recon_loss = reconstruction_loss(reconstruction, x, mask, loss_type='mse')
            lap_loss = self.model.compute_laplacian_regularization()
            loss = recon_loss + lap_loss
            
            self.log('val/recon_loss', recon_loss, prog_bar=True)
            self.log('val/loss', loss, prog_bar=True)
            
        else:  # finetune
            x, y = batch[:2]
            
            logits, _ = self(x)
            
            cls_loss = self.criterion(logits, y)
            lap_loss = self.model.compute_laplacian_regularization()
            loss = cls_loss + lap_loss
            
            # Get predictions and labels
            preds = torch.argmax(logits, dim=1)
            
            # Store for epoch-level metrics
            self.validation_step_outputs.append({
                'preds': preds.cpu(),
                'labels': y.cpu(),
                'loss': loss.item()
            })
            
            self.log('val/loss', loss, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        if self.hparams.mode == 'finetune' and len(self.validation_step_outputs) > 0:
            # Aggregate predictions
            all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
            all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
            
            # Compute metrics
            acc = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            self.log('val/accuracy', acc, prog_bar=True)
            self.log('val/precision', precision)
            self.log('val/recall', recall)
            self.log('val/f1', f1, prog_bar=True)
            
            # Clear outputs
            self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        
        logits, _ = self(x)
        
        cls_loss = self.criterion(logits, y)
        lap_loss = self.model.compute_laplacian_regularization()
        loss = cls_loss + lap_loss
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Store for epoch-level metrics
        self.test_step_outputs.append({
            'preds': preds.cpu(),
            'labels': y.cpu(),
            'loss': loss.item()
        })
        
        return loss
    
    def on_test_epoch_end(self):
        if len(self.test_step_outputs) > 0:
            # Aggregate predictions
            all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
            all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
            
            # Compute metrics
            acc = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            self.log('test/accuracy', acc)
            self.log('test/precision', precision)
            self.log('test/recall', recall)
            self.log('test/f1', f1)
            
            # Store results as attributes for external access
            self.test_results = {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': all_preds.numpy(),
                'labels': all_labels.numpy()
            }
            
            # Clear outputs
            self.test_step_outputs.clear()
            
            return self.test_results
    
    def predict_step(self, batch, batch_idx):
        """For inference on unlabeled data."""
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        
        logits, _ = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return {'predictions': preds, 'probabilities': probs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.scheduler == 'constant':
            return optimizer
        
        elif self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.total_epochs - self.hparams.warmup_epochs,
                eta_min=self.hparams.min_lr
            )
            
            if self.hparams.warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=self.hparams.warmup_epochs
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, scheduler],
                    milestones=[self.hparams.warmup_epochs]
                )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        
        elif self.hparams.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=self.hparams.min_lr
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/accuracy' if self.hparams.mode == 'finetune' else 'val/loss',
                    'interval': 'epoch'
                }
            }
        
        else:
            return optimizer

