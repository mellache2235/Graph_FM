import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class TemporalAttentionPool(nn.Module):
    """Attention-based temporal pooling to preserve temporal information."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.Tanh(),
            nn.Linear(embedding_dim // 4, 1)
        )
    
    def forward(self, x):
        # x: (batch, channels, timesteps)
        x_t = x.transpose(1, 2)  # (batch, timesteps, channels)
        attn_weights = self.attention(x_t)  # (batch, timesteps, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (x_t * attn_weights).sum(dim=1)  # (batch, channels)
        return pooled, attn_weights


class CNN1D_fMRI(nn.Module):
    def __init__(
        self,
        n_rois: int = 246,
        n_classes: int = 2,
        n_timesteps: int = 200,
        embedding_dim: int = 512,
        kernel_size: int = 3,
        mode: str = 'pretrain',
        use_laplacian: bool = False,
        use_gradient_mixing: bool = False,
        laplacian_lambda: float = 0.01,
        laplacian_layer: int = 2,
        dropout: float = 0.5,
        temporal_pool: str = 'mean'  # 'mean' or 'attention'
    ):
        super().__init__()
        
        self.n_rois = n_rois
        self.n_classes = n_classes
        self.n_timesteps = n_timesteps
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.use_laplacian = use_laplacian
        self.use_gradient_mixing = use_gradient_mixing
        self.laplacian_lambda = laplacian_lambda
        self.laplacian_layer = laplacian_layer
        self.temporal_pool = temporal_pool
        
        self.register_buffer('L', None)
        self.register_buffer('roi_order', None)
        
        if mode == 'pretrain':
            self.encoder_layers = nn.ModuleList([
                nn.Conv1d(n_rois, embedding_dim, kernel_size, padding=kernel_size//2),
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size, padding=kernel_size//2),
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size, padding=kernel_size//2),
                nn.Conv1d(embedding_dim, n_rois, kernel_size, padding=kernel_size//2),
                nn.Conv1d(n_rois, n_rois, kernel_size, padding=kernel_size//2),
                nn.Conv1d(n_rois, n_rois, kernel_size, padding=kernel_size//2),
            ])
            self.embedding_layer_idx = 2
        else:
            self.encoder_layers = nn.ModuleList([
                nn.Conv1d(n_rois, embedding_dim, kernel_size, padding=kernel_size//2),
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size, padding=kernel_size//2),
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size, padding=kernel_size//2),
            ])
            self.embedding_layer_idx = 2
            self.classifier = nn.Linear(embedding_dim, n_classes)
        
        # Gradient-aware mixing: 2D conv on ordered ROIs + MLP gating
        if use_gradient_mixing:
            self.grad_mix_conv = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0))
            self.grad_gate_mlp = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            self.register_buffer('gradient_coords', None)
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_maps = {}
        
        # Temporal pooling
        if temporal_pool == 'attention':
            self.temporal_attention = TemporalAttentionPool(embedding_dim)
        
    def set_laplacian(self, L: torch.Tensor):
        self.L = L
        
    def set_roi_order(self, order: torch.Tensor):
        self.roi_order = order
        
    def set_gradient_coords(self, coords: torch.Tensor):
        self.gradient_coords = coords
    
    def apply_gradient_mixing(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Gradient-aware mixing with mask guardrails to prevent pretraining shortcuts
        if not self.use_gradient_mixing:
            return x
        
        batch, channels, timesteps = x.shape
        
        # During pretraining with masking, apply guardrails to prevent spatial shortcuts
        if self.mode == 'pretrain' and mask is not None:
            # Mask-gated mixing: zero out features at masked positions before mixing
            # Transpose to (batch, channels, timesteps) -> (batch, timesteps, channels)
            # But x is already (batch, channels, timesteps) after conv
            # We need to prevent same-time spatial leakage
            # For now, skip gradient mixing during pretraining with masks
            return x
        
        # Apply gradient mixing for finetuning or pretraining without masks
        if self.roi_order is not None:
            x_ordered = x[:, :, self.roi_order]
            x_2d = x_ordered.view(batch * channels, 1, timesteps, 1)
            x_mixed = self.grad_mix_conv(x_2d)
            x_mixed = x_mixed.view(batch, channels, timesteps)
            x_mixed = x_mixed[:, :, torch.argsort(self.roi_order)]
        else:
            x_mixed = x
        
        if self.gradient_coords is not None:
            gates = self.grad_gate_mlp(self.gradient_coords).squeeze(-1)
            x_mixed = x_mixed * gates.view(1, 1, -1)
        
        return x_mixed
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        x = x.transpose(1, 2)
        self.hidden_maps = {}
        
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            
            if self.use_gradient_mixing and i < 2:
                x = self.apply_gradient_mixing(x, mask)
            
            if self.use_laplacian and i == self.laplacian_layer:
                self.hidden_maps['laplacian_layer'] = x
                # Store mask for mask-aware Laplacian regularization
                if mask is not None:
                    self.hidden_maps['mask'] = mask
            
            if i < len(self.encoder_layers) - 1:
                x = F.relu(x)
            
            if i == self.embedding_layer_idx:
                embedding = x
        
        if self.mode == 'pretrain':
            reconstruction = x.transpose(1, 2)
            if return_embedding:
                if self.temporal_pool == 'attention':
                    pooled_embedding, attn_weights = self.temporal_attention(embedding)
                    self.hidden_maps['attention_weights'] = attn_weights
                    return pooled_embedding, self.hidden_maps
                else:
                    return embedding.mean(dim=2), self.hidden_maps
            return reconstruction, self.hidden_maps
        else:
            if self.temporal_pool == 'attention':
                embedding, attn_weights = self.temporal_attention(embedding)
                self.hidden_maps['attention_weights'] = attn_weights
            else:
                embedding = embedding.mean(dim=2)
            
            if return_embedding:
                return embedding, self.hidden_maps
            embedding = self.dropout(embedding)
            logits = self.classifier(embedding)
            return logits, self.hidden_maps
    
    def compute_laplacian_regularization(self, lambda_scale: float = 1.0) -> torch.Tensor:
        # Laplacian smoothing: λ * tr(H^T L H) where H is hidden activations and L is graph Laplacian
        # MASK-AWARE: Only compute on unmasked positions to prevent spatial shortcuts during pretraining
        # lambda_scale: warmup multiplier (0→1 over first few epochs)
        if not self.use_laplacian or self.L is None or 'laplacian_layer' not in self.hidden_maps:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        H = self.hidden_maps['laplacian_layer']  # (batch, channels, timesteps)
        mask = self.hidden_maps.get('mask', None)  # (batch, timesteps) if present
        batch, channels, timesteps = H.shape
        
        if channels != self.n_rois:
            return torch.tensor(0.0, device=H.device)
        
        # Transpose to (batch, timesteps, n_rois) for easier processing
        H_bt = H.transpose(1, 2)  # (batch, timesteps, n_rois)
        
        if mask is not None:
            # MASK-AWARE REGULARIZATION: Only penalize on unmasked (context) positions
            # mask: 1 = masked, 0 = keep
            ctx = (1 - mask).float().unsqueeze(-1)  # (batch, timesteps, 1), 1 = context
            H_ctx = H_bt * ctx  # Zero out masked positions
            
            # Compute smoothness only on context: tr(H_ctx^T L H_ctx)
            # H_ctx: (batch, timesteps, n_rois), L: (n_rois, n_rois)
            # smooth = sum_b,t H_ctx[b,t,:] @ L @ H_ctx[b,t,:]
            smooth = torch.einsum('btn,nm,btm->', H_ctx, self.L, H_ctx)
            ctx_count = ctx.sum() + 1e-8
            reg_loss = smooth / ctx_count
        else:
            # No mask: average over all timesteps
            smooth = torch.einsum('btn,nm,btm->', H_bt, self.L, H_bt)
            reg_loss = smooth / (batch * timesteps)
        
        return self.laplacian_lambda * lambda_scale * reg_loss
    
    def load_pretrained_encoder(self, pretrained_path: str, freeze: bool = True):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        encoder_state = {k.replace('encoder_layers.', ''): v 
                        for k, v in checkpoint.items() 
                        if k.startswith('encoder_layers')}
        self.encoder_layers.load_state_dict(encoder_state, strict=False)
        if freeze:
            for param in self.encoder_layers.parameters():
                param.requires_grad = False

