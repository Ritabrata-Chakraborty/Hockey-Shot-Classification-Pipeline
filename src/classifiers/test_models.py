#!/usr/bin/env python3
"""
Test trained models on batch or single trajectories.
Generates visualizations with actual vs predicted labels.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import csv
from collections import defaultdict

import torch
from .tcn_train_test import TCNTrainTest
from .ml_trainer import MLTrainer
from ..physics.shot_logical import TrajectoryPoint


def load_test_data(data_path, num_samples=None):
    """Load test trajectories from CSV."""
    
    # Handle glob patterns
    data_files = glob.glob(data_path)
    if not data_files:
        raise FileNotFoundError(f"No data files found matching: {data_path}")
    
    # Sort by modification time (most recent first)
    data_files.sort(key=os.path.getmtime, reverse=True)
    data_file = data_files[0]
    
    print(f"Loading data from: {data_file}")
    
    trajectories = defaultdict(list)
    current_shot = {}
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['shot_type'], int(row['shot_id']))
            if key not in current_shot:
                current_shot[key] = []
            current_shot[key].append(TrajectoryPoint(
                float(row['time']),
                float(row['x']),
                float(row['y']),
                float(row['z'])
            ))
    
    # Group by shot type
    for (shot_type, shot_id), points in current_shot.items():
        trajectories[shot_type].append(points)
    
    # Limit samples if specified
    if num_samples:
        limited = {}
        for shot_type, traj_list in trajectories.items():
            limited[shot_type] = traj_list[:num_samples]
        trajectories = limited
    
    total = sum(len(v) for v in trajectories.values())
    print(f"✓ Loaded {total} trajectories from {len(trajectories)} shot types")
    
    return trajectories


def test_single_trajectory(model_path, model_type, trajectory, true_label, output_path):
    """Test single trajectory and generate visualization."""
    
    # Load model
    if model_type == 'tcn':
        model = TCNTrainTest.load(model_path, device='cpu')
        model.test_single(trajectory, true_label, save_path=output_path)
    else:
        model = MLTrainer.load(model_path)
        model.test_single(trajectory, true_label, save_path=output_path)
    
    print(f"✓ Test complete: {output_path}")


def test_batch(model_path, model_type, trajectories, labels, output_dir):
    """Test batch of trajectories and generate summary."""
    
    print(f"\nTesting {model_type.upper()} on {len(trajectories)} trajectories...")
    
    # Load model
    if model_type == 'tcn':
        model = TCNTrainTest.load(model_path, device='cpu')
    else:
        model = MLTrainer.load(model_path)
    
    # Test
    result = model.test_batch(trajectories, labels)
    if len(result) == 3:
        acc, cm, f1 = result
    else:
        acc, cm = result
    
    # Generate report
    os.makedirs(output_dir, exist_ok=True)
    
    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    
    shot_types = ['hit', 'slap', 'push_flick', 'drag_flick']
    im = ax.imshow(cm, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Labels
    ax.set_xticks(np.arange(len(shot_types)))
    ax.set_yticks(np.arange(len(shot_types)))
    ax.set_xticklabels(shot_types)
    ax.set_yticklabels(shot_types)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(shot_types)):
        for j in range(len(shot_types)):
            text = ax.text(j, i, int(cm[i, j]),
                          ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=14, fontweight='bold')
    
    ax.set_title(f'{model_type.upper()} - Test Accuracy: {acc:.2%}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    
    plt.tight_layout()
    
    output_path = f'{output_dir}/{model_type}_test_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Accuracy: {acc:.2%}")
    print(f"✓ Results saved to: {output_path}")
    
    return acc, cm


def test_all_models(models_dir, data_path, output_dir, num_samples=None):
    """Test all trained models."""
    
    print("="*70)
    print(" "*20 + "MODEL TESTING")
    print("="*70)
    
    # Load data
    trajectories_dict = load_test_data(data_path, num_samples)
    
    # Flatten to lists
    all_traj = []
    all_labels = []
    for shot_type, traj_list in trajectories_dict.items():
        all_traj.extend(traj_list)
        all_labels.extend([shot_type] * len(traj_list))
    
    print(f"\nTotal test samples: {len(all_traj)}")
    print()
    
    # Find models
    models = {
        'tcn': f'{models_dir}/best_tcn_model.pth',
        'xgboost': f'{models_dir}/xgboost_model.pkl',
        'svm': f'{models_dir}/svm_model.pkl',
        'random_forest': f'{models_dir}/random_forest_model.pkl',
        'knn': f'{models_dir}/knn_model.pkl'
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"⚠️  Skipping {model_name}: model not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*70}")
        
        try:
            acc, cm = test_batch(model_path, model_name, all_traj, all_labels, output_dir)
            results[model_name] = {'accuracy': acc, 'confusion_matrix': cm}
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(" "*20 + "SUMMARY")
    print(f"{'='*70}")
    
    if results:
        print(f"\n{'Model':<15s} {'Accuracy':<15s}")
        print("-"*30)
        for model_name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{model_name.upper():<15s} {result['accuracy']:<15.2%}")
        
        print(f"\n✓ Test results saved to: {output_dir}/")
        print(f"{'='*70}\n")
    else:
        print("\n❌ No models were successfully tested")


def main():
    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--mode', choices=['single', 'batch'], default='batch',
                       help='Test mode: single trajectory or batch')
    parser.add_argument('--model-path', type=str, 
                       help='Path to model file (for single mode)')
    parser.add_argument('--model-type', type=str, 
                       choices=['tcn', 'xgboost', 'svm', 'random_forest', 'knn'],
                       help='Model type (for single mode)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--output', type=str, default='models/batch_test_results',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples per class (default: all)')
    parser.add_argument('--models-dir', type=str, default='models/checkpoints',
                       help='Directory with trained models (for batch mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.model_path or not args.model_type:
            print("❌ For single mode, --model-path and --model-type are required")
            sys.exit(1)
        
        # Load one trajectory
        trajectories_dict = load_test_data(args.data, num_samples=1)
        shot_type = list(trajectories_dict.keys())[0]
        trajectory = trajectories_dict[shot_type][0]
        
        output_path = f'{args.output}/{args.model_type}_single_test.png'
        os.makedirs(args.output, exist_ok=True)
        
        test_single_trajectory(args.model_path, args.model_type, 
                              trajectory, shot_type, output_path)
    else:
        # Batch mode
        test_all_models(args.models_dir, args.data, args.output, args.num_samples)


if __name__ == '__main__':
    main()

