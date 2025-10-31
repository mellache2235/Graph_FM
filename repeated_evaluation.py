"""
Repeated evaluation with 50x train-valid-test splits (64-16-20)
No early stopping - trains for full epoch count
Tracks: accuracy, macro-precision, macro-recall, macro-f1
"""

import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm

from model_lightning import LitfMRIModel
from data_loader import fMRIDataset, load_data_for_laplacian, load_data_for_gradients
from utils import build_laplacian_from_data, compute_principal_gradient


def create_split_loaders(
    dataset: fMRIDataset,
    split_seed: int,
    batch_size: int = 32,
    num_workers: int = 4
):
    """
    Create train-valid-test loaders with 64-16-20 split.
    
    Args:
        dataset: Full dataset
        split_seed: Random seed for split
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    # First split: 80% train+val, 20% test
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=split_seed,
        stratify=dataset.Y
    )
    
    # Second split: 64% train, 16% val (from the 80%)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.20,  # 16% of total (20% of 80%)
        random_state=split_seed,
        stratify=dataset.Y[train_val_indices]
    )
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader, train_indices


def run_single_split(
    split_idx: int,
    dataset: fMRIDataset,
    args: argparse.Namespace,
    laplacian: torch.Tensor = None,
    roi_order: torch.Tensor = None,
    gradient_coords: torch.Tensor = None
):
    """
    Run training and evaluation for a single split.
    
    Returns:
        Dictionary with metrics
    """
    print(f"\n{'='*80}")
    print(f"SPLIT {split_idx + 1}/{args.n_repeats}")
    print(f"{'='*80}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, train_indices = create_split_loaders(
        dataset,
        split_seed=args.seed + split_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Split sizes - Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    model = LitfMRIModel(
        n_rois=args.n_rois,
        n_classes=args.n_classes,
        embedding_dim=args.embedding_dim,
        mode='finetune',
        use_laplacian=args.use_laplacian,
        use_gradient_mixing=args.use_gradient_mixing,
        laplacian_lambda=args.laplacian_lambda,
        laplacian_layer=args.laplacian_layer,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs
    )
    
    # Load pretrained weights if provided
    if args.pretrained_model is not None:
        print(f"Loading pretrained weights...")
        pretrained_state = torch.load(args.pretrained_model, map_location='cpu')
        model_dict = model.model.state_dict()
        pretrained_dict = {}
        
        for i in range(3):
            for param_name in ['weight', 'bias']:
                key = f'encoder_layers.{i}.{param_name}'
                if key in pretrained_state:
                    pretrained_dict[key] = pretrained_state[key]
        
        model_dict.update(pretrained_dict)
        model.model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} pretrained parameters")
    
    # Set graph regularization
    if laplacian is not None:
        model.set_laplacian(laplacian)
    if roi_order is not None:
        model.set_roi_order(roi_order)
    if gradient_coords is not None:
        model.set_gradient_coords(gradient_coords)
    
    # Setup callbacks - NO early stopping
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, f'split_{split_idx}'),
        filename='best',
        monitor='val/accuracy',
        mode='max',
        save_top_k=1
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        deterministic=True
    )
    
    # Train - NO early stopping, runs full epoch count
    print(f"\nTraining for {args.epochs} epochs (no early stopping)...")
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model based on validation accuracy
    best_model = LitfMRIModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        n_rois=args.n_rois,
        n_classes=args.n_classes,
        embedding_dim=args.embedding_dim
    )
    
    # Set graph regularization for loaded model
    if laplacian is not None:
        best_model.set_laplacian(laplacian)
    if roi_order is not None:
        best_model.set_roi_order(roi_order)
    if gradient_coords is not None:
        best_model.set_gradient_coords(gradient_coords)
    
    # Test
    print("\nEvaluating on test set...")
    trainer.test(best_model, test_loader)
    
    results = best_model.test_results
    
    print(f"\nResults - Acc: {results['accuracy']:.4f}, "
          f"Prec: {results['precision']:.4f}, "
          f"Rec: {results['recall']:.4f}, "
          f"F1: {results['f1']:.4f}")
    
    return results


def repeated_evaluation(args):
    """
    Run repeated evaluation with multiple train-valid-test splits.
    """
    # Set seeds for reproducibility
    pl.seed_everything(args.seed)
    
    # Load full dataset
    print("\nLoading dataset...")
    dataset = fMRIDataset(
        args.data_path,
        load_ids=False,
        source_tr=args.source_tr,
        target_tr=args.target_tr if args.source_tr is not None else None
    )
    
    data_info = dataset.get_data_info()
    print(f"Dataset: {data_info['n_subjects']} subjects, "
          f"{data_info['n_timesteps']} timesteps, "
          f"{data_info['n_rois']} ROIs, "
          f"{data_info['n_classes']} classes")
    
    # Setup graph regularization (computed once on full dataset)
    laplacian = None
    roi_order = None
    gradient_coords = None
    
    if args.use_laplacian:
        print("\nComputing Laplacian matrix...")
        train_data = load_data_for_laplacian(args.data_path, max_subjects=500)
        laplacian = build_laplacian_from_data(
            train_data,
            top_k=args.laplacian_top_k,
            normalized=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Laplacian shape: {laplacian.shape}")
    
    if args.use_gradient_mixing:
        print("\nComputing principal gradients...")
        train_data = load_data_for_gradients(args.data_path, max_subjects=500)
        gradient_coords_np, roi_order_np = compute_principal_gradient(train_data, n_components=3)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        roi_order = torch.from_numpy(roi_order_np).long().to(device)
        gradient_coords = torch.from_numpy(gradient_coords_np).float().to(device)
        print(f"Gradient coordinates shape: {gradient_coords.shape}")
    
    # Run repeated evaluations
    print(f"\n{'='*80}")
    print(f"RUNNING {args.n_repeats} REPEATED EVALUATIONS")
    print(f"{'='*80}")
    
    all_results = []
    
    for split_idx in range(args.n_repeats):
        results = run_single_split(
            split_idx,
            dataset,
            args,
            laplacian=laplacian,
            roi_order=roi_order,
            gradient_coords=gradient_coords
        )
        
        all_results.append({
            'split': split_idx,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1']
        })
    
    # Aggregate results
    df_results = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS ACROSS ALL SPLITS")
    print("=" * 80)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        mean = df_results[metric].mean()
        std = df_results[metric].std()
        print(f"{metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")
    
    # Save results
    results_file = os.path.join(args.save_dir, 'repeated_evaluation_results.csv')
    df_results.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save summary
    summary_file = os.path.join(args.save_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Repeated Evaluation Summary ({args.n_repeats} splits)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Splits: {args.n_repeats}x train-valid-test (64-16-20)\n")
        f.write(f"Epochs per split: {args.epochs} (no early stopping)\n\n")
        f.write("Results (mean ± std):\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            mean = df_results[metric].mean()
            std = df_results[metric].std()
            f.write(f"  {metric.capitalize():12s}: {mean:.4f} ± {std:.4f}\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(description='Repeated evaluation with train-valid-test splits')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to .pklz data file')
    parser.add_argument('--n_repeats', type=int, default=50, help='Number of repeated evaluations')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    
    # TR resampling
    parser.add_argument('--source_tr', type=float, default=None, help='Source TR')
    parser.add_argument('--target_tr', type=float, default=0.72, help='Target TR')
    
    # Model arguments
    parser.add_argument('--n_rois', type=int, default=246, help='Number of ROIs')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Pretrained model
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model')
    
    # Regularization
    parser.add_argument('--use_laplacian', action='store_true', help='Use Laplacian smoothing')
    parser.add_argument('--laplacian_lambda', type=float, default=0.01, help='Laplacian weight')
    parser.add_argument('--laplacian_layer', type=int, default=2, help='Laplacian layer')
    parser.add_argument('--laplacian_top_k', type=int, default=10, help='Top-k for adjacency')
    parser.add_argument('--use_gradient_mixing', action='store_true', help='Use gradient mixing')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (no early stopping)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['constant', 'cosine', 'plateau'])
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='repeated_eval_results', help='Save directory')
    
    args = parser.parse_args()
    
    # Validate learning rate
    if args.lr > args.max_lr:
        args.lr = args.max_lr
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("REPEATED EVALUATION CONFIGURATION")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("=" * 80)
    
    # Run evaluation
    results_df = repeated_evaluation(args)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

