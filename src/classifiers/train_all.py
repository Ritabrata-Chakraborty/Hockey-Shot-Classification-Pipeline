"""
Train all models (TCN + Classical ML) and generate comprehensive comparison report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .tcn_train_test import TCNTrainTest
from .ml_trainer import MLTrainer
import time
import os
import sys
import csv
import glob
from collections import defaultdict
from datetime import datetime
from ..physics.shot_logical import TrajectoryPoint


def train_all_models(data_path, output_dir='models/checkpoints', epochs=150):
    """
    Train all 5 models and generate comprehensive comparison report.
    
    Args:
        data_path: Path to CSV data file (can use wildcards)
        output_dir: Directory to save results
        epochs: Number of epochs for TCN training (default: 150)
    
    Models:
    - TCN (temporal convolutional network)
    - XGBoost (gradient boosting)
    - Random Forest (ensemble)
    - SVM (support vector machine)
    - KNN (k-nearest neighbors)
    
    Generates:
    - Comprehensive comparison table (train/val/test accuracy, F1 scores)
    - Individual confusion matrices for each model
    - Training time and performance comparison
    - Detailed accuracy plots
    - Summary statistics
    """
    print("="*80)
    print(" " * 20 + "FULL PIPELINE TRAINING")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    print("="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    data_files = sorted(glob.glob(data_path), key=os.path.getmtime, reverse=True)
    if not data_files:
        print(f"❌ No data files found matching: {data_path}")
        sys.exit(1)
    
    # Use most recent file
    data_file = data_files[0]
    file_size_mb = os.path.getsize(data_file) / (1024 * 1024)
    print(f"Loading from: {data_file}")
    print(f"File size: {file_size_mb:.1f} MB (modified: {datetime.fromtimestamp(os.path.getmtime(data_file))})")
    
    trajectories, current_shot = defaultdict(list), {}
    with open(data_file, 'r') as f:
        for row in csv.DictReader(f):
            key = (row['shot_type'], int(row['shot_id']))
            if key not in current_shot:
                current_shot[key] = []
            current_shot[key].append(TrajectoryPoint(
                float(row['time']), float(row['x']), 
                float(row['y']), float(row['z'])))
    
    for (shot_type, _), points in current_shot.items():
        trajectories[shot_type].append(sorted(points, key=lambda p: p.time))
    
    all_traj, all_labels = [], []
    for shot_type, traj_list in trajectories.items():
        all_traj.extend(traj_list)
        all_labels.extend([shot_type] * len(traj_list))
    
    print(f"✅ Loaded {len(all_traj)} trajectories")
    print(f"✅ Shot types: {set(all_labels)}")
    print(f"✅ Samples per type: {dict((k, all_labels.count(k)) for k in set(all_labels))}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train all models
    print(f"\n{'='*80}")
    print("STEP 2: TRAINING ALL MODELS")
    print("="*80)
    print("Models to train: TCN, XGBoost, Random Forest, SVM, KNN")
    print()
    
    results = {}
    models_config = [
        ('tcn', 'Temporal Convolutional Network'),
        ('xgboost', 'XGBoost Classifier'),
        ('random_forest', 'Random Forest'),
        ('svm', 'Support Vector Machine'),
        ('knn', 'K-Nearest Neighbors')
    ]
    
    for idx, (model_name, model_desc) in enumerate(models_config, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {idx}/5: {model_desc} ({model_name.upper()})")
        print('='*80)
        
        start_time = time.time()
        
        try:
            if model_name == 'tcn':
                trainer = TCNTrainTest()
                metrics = trainer.train(all_traj, all_labels, save_dir=output_dir, num_epochs=epochs)
            else:
                trainer = MLTrainer(model_type=model_name)
                metrics = trainer.train(all_traj, all_labels)
                # Save classical ML model
                model_path = f'{output_dir}/{model_name}_model.pkl'
                trainer.save(model_path)
                print(f"   Model saved: {model_path}")
            
            training_time = time.time() - start_time
            
            results[model_name] = {
                'train_acc': metrics['train_acc'],
                'val_acc': metrics['val_acc'],
                'test_acc': metrics['test_acc'],
                'confusion_matrix': metrics['confusion_matrix'],
                'f1_score': metrics.get('f1_score', 0.0),
                'training_time': training_time
            }
            
            print(f"\n✅ Training completed successfully!")
            print(f"   Train Acc:  {metrics['train_acc']*100:.2f}%")
            print(f"   Val Acc:    {metrics['val_acc']*100:.2f}%")
            print(f"   Test Acc:   {metrics['test_acc']*100:.2f}%")
            print(f"   F1 Score:   {metrics.get('f1_score', 0.0):.4f}")
            print(f"   Time:       {training_time:.1f}s")
            
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                'train_acc': 0.0,
                'val_acc': 0.0,
                'test_acc': 0.0,
                'confusion_matrix': np.zeros((len(set(all_labels)), len(set(all_labels)))),
                'f1_score': 0.0,
                'training_time': 0.0,
                'error': str(e)
            }
    
    # Generate comprehensive comparison report
    print(f"\n{'='*80}")
    print("STEP 3: GENERATING COMPARISON REPORT")
    print('='*80)
    
    # Filter out failed models
    all_models = [name for name, _ in models_config]
    successful_models = [m for m in all_models if 'error' not in results[m]]
    failed_models = [m for m in all_models if 'error' in results[m]]
    
    if failed_models:
        print(f"⚠️  Failed models: {', '.join(failed_models)}")
    
    if not successful_models:
        print("❌ No models trained successfully!")
        return results
    
    print(f"✅ Successfully trained: {', '.join([m.upper() for m in successful_models])}")
    
    # Create comprehensive comparison table
    df = pd.DataFrame({
        'Model': [m.upper() for m in successful_models],
        'Train Acc (%)': [results[m]['train_acc']*100 for m in successful_models],
        'Val Acc (%)': [results[m]['val_acc']*100 for m in successful_models],
        'Test Acc (%)': [results[m]['test_acc']*100 for m in successful_models],
        'F1 Score': [results[m]['f1_score'] for m in successful_models],
        'Time (s)': [results[m]['training_time'] for m in successful_models]
    })
    
    # Sort by test accuracy (descending)
    df = df.sort_values('Test Acc (%)', ascending=False).reset_index(drop=True)
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS - SUMMARY TABLE")
    print('='*80)
    print(df.to_string(index=False))
    print('='*80)
    
    # Save table
    csv_path = f'{output_dir}/comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Comparison table saved to {csv_path}")
    
    # Generate comprehensive plots
    _plot_comprehensive_comparison(df, results, successful_models, output_dir)
    
    # Generate individual model reports
    _generate_model_reports(results, successful_models, output_dir)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    best_idx = df['Test Acc (%)'].idxmax()
    best_model = df.loc[best_idx, 'Model']
    best_acc = df.loc[best_idx, 'Test Acc (%)']
    best_f1 = df.loc[best_idx, 'F1 Score']
    best_time = df.loc[best_idx, 'Time (s)']
    
    print(f"🏆 Best Model: {best_model}")
    print(f"   Test Accuracy: {best_acc:.2f}%")
    print(f"   F1 Score:      {best_f1:.4f}")
    print(f"   Training Time: {best_time:.1f}s")
    
    print(f"\n📊 Average Performance:")
    print(f"   Mean Test Acc: {df['Test Acc (%)'].mean():.2f}%")
    print(f"   Mean F1 Score: {df['F1 Score'].mean():.4f}")
    print(f"   Total Time:    {df['Time (s)'].sum():.1f}s")
    
    print(f"\n📁 Output Files:")
    print(f"   {output_dir}/comparison_table.csv")
    print(f"   {output_dir}/comprehensive_comparison.png")
    print(f"   {output_dir}/confusion_matrices_all.png")
    print(f"   {output_dir}/model_summary.txt")
    print(f"   {output_dir}/<model>_model.pth or .pkl")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print('='*80)
    
    return results


def _plot_comprehensive_comparison(df, results, successful_models, output_dir):
    """Generate comprehensive comparison plots."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy comparison (grouped bar chart)
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax1.bar(x - width, df['Train Acc (%)'], width, label='Train', color='skyblue', edgecolor='black')
    bars2 = ax1.bar(x, df['Val Acc (%)'], width, label='Val', color='lightcoral', edgecolor='black')
    bars3 = ax1.bar(x + width, df['Test Acc (%)'], width, label='Test', color='lightgreen', edgecolor='black')
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison (Train / Val / Test)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Training time comparison
    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.barh(df['Model'], df['Time (s)'], color='orange', edgecolor='black')
    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Training Time', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3, axis='x')
    
    for bar, time_val in zip(bars, df['Time (s)']):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{time_val:.1f}s', ha='left', va='center', fontsize=9)
    
    # 3. F1 Score comparison
    ax3 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    bars = ax3.bar(df['Model'], df['F1 Score'], color=colors, edgecolor='black')
    ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax3.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.05])
    
    for bar, f1_val in zip(bars, df['F1 Score']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Test Accuracy vs Training Time scatter
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(df['Time (s)'], df['Test Acc (%)'], 
                         s=200, c=df['F1 Score'], cmap='viridis', 
                         edgecolors='black', linewidths=2)
    
    for idx, model in enumerate(df['Model']):
        ax4.annotate(model, (df['Time (s)'].iloc[idx], df['Test Acc (%)'].iloc[idx]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Training Time (s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Accuracy vs Time (colored by F1)', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('F1 Score', fontsize=10)
    
    # 5. Performance summary text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    summary_text = [
        "PERFORMANCE SUMMARY",
        "=" * 30,
        f"",
        f"Best Model:",
        f"  {df.loc[0, 'Model']}",
        f"  Acc: {df.loc[0, 'Test Acc (%)']:.2f}%",
        f"  F1:  {df.loc[0, 'F1 Score']:.4f}",
        f"",
        f"Fastest Training:",
        f"  {df.loc[df['Time (s)'].idxmin(), 'Model']}",
        f"  Time: {df['Time (s)'].min():.1f}s",
        f"",
        f"Average Results:",
        f"  Acc: {df['Test Acc (%)'].mean():.2f}%",
        f"  F1:  {df['F1 Score'].mean():.4f}",
        f"  Time: {df['Time (s)'].mean():.1f}s",
    ]
    
    y_pos = 0.95
    for line in summary_text:
        fontweight = 'bold' if line.startswith('PERFORMANCE') or line.startswith('Best') or line.startswith('Fastest') or line.startswith('Average') else 'normal'
        fontsize = 11 if fontweight == 'bold' else 10
        ax5.text(0.1, y_pos, line, transform=ax5.transAxes,
                fontsize=fontsize, verticalalignment='top', fontweight=fontweight,
                family='monospace')
        y_pos -= 0.06
    
    plt.savefig(f'{output_dir}/comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Comprehensive comparison plots saved to {output_dir}/comprehensive_comparison.png")
    
    # Plot all confusion matrices
    _plot_all_confusion_matrices(results, successful_models, output_dir)


def _plot_all_confusion_matrices(results, successful_models, output_dir):
    """Plot all confusion matrices in a grid."""
    n_models = len(successful_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Convert to array for consistent indexing
    if n_models == 1:
        axes = [axes]  # Single element list
    else:
        axes = axes.flatten()  # Flatten multi-dimensional array
    
    for idx, model in enumerate(successful_models):
        ax = axes[idx]
        cm = results[model]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_title(f'{model.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ All confusion matrices saved to {output_dir}/confusion_matrices_all.png")


def _generate_model_reports(results, successful_models, output_dir):
    """Generate text summary report."""
    report_path = f'{output_dir}/model_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HOCKEY SHOT CLASSIFICATION - MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models Trained: {len(successful_models)}\n")
        f.write("="*80 + "\n\n")
        
        for model in successful_models:
            r = results[model]
            f.write(f"{'='*80}\n")
            f.write(f"MODEL: {model.upper()}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Train Accuracy:      {r['train_acc']*100:.2f}%\n")
            f.write(f"Validation Accuracy: {r['val_acc']*100:.2f}%\n")
            f.write(f"Test Accuracy:       {r['test_acc']*100:.2f}%\n")
            f.write(f"F1 Score:            {r['f1_score']:.4f}\n")
            f.write(f"Training Time:       {r['training_time']:.2f} seconds\n")
            f.write(f"\nConfusion Matrix:\n")
            f.write(np.array2string(r['confusion_matrix'], separator=', '))
            f.write(f"\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Model summary report saved to {report_path}")
    
    return results


# CLI support
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all models (TCN + 4 classical ML)')
    parser.add_argument('data_path', help='Path to CSV data file (can use wildcards)')
    parser.add_argument('output_dir', nargs='?', default='model_comparison', 
                       help='Output directory for results (default: model_comparison)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs for TCN training (default: 150)')
    
    args = parser.parse_args()
    
    train_all_models(args.data_path, args.output_dir, args.epochs)

