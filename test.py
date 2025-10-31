"""
Test script for applying trained model to external datasets
"""

import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from model_lightning import LitfMRIModel
from data_loader import create_data_loader


def test_model(
    model_path: str,
    test_data_path: str,
    n_rois: int = 246,
    n_classes: int = 2,
    embedding_dim: int = 512,
    batch_size: int = 32,
    source_tr: float = None,
    target_tr: float = 0.72,
    device: str = 'auto',
    track_ids: bool = False
):
    """
    Test a trained model on external dataset.
    
    Args:
        model_path: Path to trained model checkpoint
        test_data_path: Path to test data .pklz file
        n_rois: Number of ROIs
        n_classes: Number of classes
        embedding_dim: Embedding dimension
        batch_size: Batch size
        source_tr: Source TR for test data (if different from training)
        target_tr: Target TR to resample to
        device: Device to use ('auto', 'gpu', 'cpu')
        track_ids: Whether to track subject IDs
    
    Returns:
        Dictionary with test metrics and predictions
    """
    # Create test data loader
    test_loader = create_data_loader(
        test_data_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        load_ids=track_ids,
        pretraining=False,
        source_tr=source_tr,
        target_tr=target_tr if source_tr is not None else None
    )
    
    # Load model
    print(f"\nLoading model from {model_path}")
    model = LitfMRIModel.load_from_checkpoint(
        model_path,
        n_rois=n_rois,
        n_classes=n_classes,
        embedding_dim=embedding_dim
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator=device,
        devices=1 if device != 'cpu' else None,
        logger=False
    )
    
    # Test
    print("\nRunning test evaluation...")
    trainer.test(model, test_loader)
    
    # Get results
    results = model.test_results
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test fMRI model on external dataset')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data .pklz file')
    
    # Model architecture
    parser.add_argument('--n_rois', type=int, default=246, help='Number of ROIs')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    
    # Data arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--source_tr', type=float, default=None, help='Source TR of test data')
    parser.add_argument('--target_tr', type=float, default=0.72, help='Target TR to resample to')
    parser.add_argument('--track_ids', action='store_true', help='Track subject IDs')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'gpu', 'cpu'],
                       help='Device to use')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("TEST CONFIGURATION")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("=" * 80)
    
    # Run test
    results = test_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        n_rois=args.n_rois,
        n_classes=args.n_classes,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        source_tr=args.source_tr,
        target_tr=args.target_tr,
        device=args.device,
        track_ids=args.track_ids
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f} (macro)")
    print(f"Recall:    {results['recall']:.4f} (macro)")
    print(f"F1 Score:  {results['f1']:.4f} (macro)")
    print("=" * 80)
    
    # Confusion matrix
    cm = confusion_matrix(results['labels'], results['predictions'])
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(results['labels'], results['predictions']))
    
    # Save results
    output_file = os.path.join(args.output_dir, 'test_results.npz')
    np.savez(
        output_file,
        accuracy=results['accuracy'],
        precision=results['precision'],
        recall=results['recall'],
        f1=results['f1'],
        predictions=results['predictions'],
        labels=results['labels'],
        confusion_matrix=cm
    )
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

