
"""
TCN training and testing with comprehensive visualization.
Supports both batch testing and single-shot testing with trajectory visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Tuple, Dict
import os
import sys
import csv
import glob
from collections import defaultdict
from datetime import datetime

from .tcn_model import HockeyTCNClassifier, TrajectoryDataset, TCNTrainingConfig, create_tcn_model
from ..physics.shot_logical import TrajectoryPoint, get_all_shot_types


class TCNTrainTest:
    """Comprehensive TCN training and testing with visualization."""
    
    def __init__(self, config=None):
        self.config = config or TCNTrainingConfig()
        self.model = None
        self.label_encoder = LabelEncoder()
        # Two scalers: one for temporal (600), one for auxiliary (52)
        self.scaler = {
            'temporal': StandardScaler(),
            'auxiliary': StandardScaler()
        }
        self.device = self.config.device
        self.shot_types = get_all_shot_types()
        self.label_encoder.fit(self.shot_types)
        
        # Training history
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []
    
    def train(self, trajectories, labels, val_size=0.2, save_dir='models/checkpoints', num_epochs=None):
        """
        Train TCN model with comprehensive logging and visualization.
        
        Args:
            trajectories: List of trajectory point lists
            labels: List of shot type labels
            val_size: Validation split ratio
            save_dir: Directory to save model checkpoints
            
        Returns:
            dict with train_acc, val_acc, test_acc, confusion_matrix, f1_score
        """
        print(f"{'='*70}")
        print(f"TRAINING TCN MODEL")
        print(f"{'='*70}")
        print(f"Total trajectories: {len(trajectories)}")
        print(f"Shot types: {self.shot_types}")
        print(f"Device: {self.device}")
        
        # Use provided num_epochs or default from config
        epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        
        os.makedirs(save_dir, exist_ok=True)
        # Training results go to separate directory
        training_results_dir = 'models/training_results'
        os.makedirs(training_results_dir, exist_ok=True)
        
        # Split data: 80% train, 10% val, 10% test
        train_traj, temp_traj, train_labels, temp_labels = train_test_split(
            trajectories, labels, test_size=0.2, random_state=42, stratify=labels)
        val_traj, test_traj, val_labels, test_labels = train_test_split(
            temp_traj, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
        
        print(f"\nData split:")
        print(f"  Train: {len(train_traj)} samples")
        print(f"  Val:   {len(val_traj)} samples")
        print(f"  Test:  {len(test_traj)} samples")
        
        # Fit scalers on training data sample
        print("\nFitting scalers on training data...")
        sample_dataset = TrajectoryDataset(train_traj[:min(100, len(train_traj))], 
                                          train_labels[:min(100, len(train_labels))],
                                          self.label_encoder, scaler=None)
        temporal_seq, auxiliary_feat = sample_dataset.sequences
        
        # Fit temporal scaler [N, 3, 200] -> [N, 600]
        temporal_flat = temporal_seq.reshape(len(temporal_seq), -1)
        self.scaler['temporal'].fit(temporal_flat)
        
        # Fit auxiliary scaler [N, 52]
        self.scaler['auxiliary'].fit(auxiliary_feat)
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = TrajectoryDataset(train_traj, train_labels, self.label_encoder, self.scaler)
        val_dataset = TrajectoryDataset(val_traj, val_labels, self.label_encoder, self.scaler)
        test_dataset = TrajectoryDataset(test_traj, test_labels, self.label_encoder, self.scaler)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Create model
        print("Creating model...")
        self.model = create_tcn_model(self.config, len(self.shot_types))
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
        
        # Training setup with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        optimizer = optim.AdamW(self.model.parameters(), 
                               lr=self.config.learning_rate, 
                               weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', 
            factor=self.config.lr_factor, 
            patience=self.config.lr_patience)
        
        # Training loop
        print(f"\n{'='*70}")
        print("TRAINING PROGRESS")
        print(f"{'='*70}")
        print(f"Training for {self.config.num_epochs} epochs with improvements:")
        print("  ✓ Label smoothing (ε=0.1)")
        print("  ✓ Gradient clipping (max_norm=1.0)")
        print("  ✓ Dilated convolutions (RF~200)")
        print("  ✓ Temporal alignment around apex")
        print("  ✓ Per-axis normalization")
        print("  ✓ Attention-based fusion")
        
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for (temporal, auxiliary), target in train_loader:
                temporal = temporal.to(self.device)
                auxiliary = auxiliary.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(temporal, auxiliary)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += (pred == target).sum().item()
                train_total += target.size(0)
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validate
            val_loss, val_acc = self._evaluate_loss_acc(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save history
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Track best model (but keep training)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                # Save best model checkpoint
                self.save(f'{save_dir}/best_tcn_model.pth')
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                      f"Best: {best_val_acc*100:.2f}% | LR: {current_lr:.2e}")
        
        print(f"\n{'='*70}")
        print(f"Training completed - {epochs} epochs!")
        print(f"Best validation accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch+1}")
        
        # Load best model for final evaluation
        self.load(f'{save_dir}/best_tcn_model.pth', self.device)
        
        # Final evaluation
        print(f"\n{'='*70}")
        print("FINAL EVALUATION")
        print(f"{'='*70}")
        
        train_acc = self._evaluate(train_loader)
        val_acc = self._evaluate(val_loader)
        test_acc, test_cm, test_f1 = self._evaluate_with_metrics(test_loader)
        
        print(f"Train Accuracy: {train_acc*100:.2f}%")
        print(f"Val Accuracy:   {val_acc*100:.2f}%")
        print(f"Test Accuracy:  {test_acc*100:.2f}%")
        print(f"Test F1 Score:  {test_f1:.4f}")
        
        # Save training curves
        self._plot_training_curves(training_results_dir)
        
        # Save confusion matrix
        self._plot_confusion_matrix(test_cm, training_results_dir)
        
        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'confusion_matrix': test_cm,
            'f1_score': test_f1,
            'best_epoch': best_epoch,
            'train_history': {
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'val_losses': self.val_losses,
                'val_accs': self.val_accs
            }
        }
    
    def _evaluate(self, dataloader):
        """Evaluate model accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (temporal, auxiliary), target in dataloader:
                temporal = temporal.to(self.device)
                auxiliary = auxiliary.to(self.device)
                target = target.to(self.device)
                output = self.model(temporal, auxiliary)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return correct / total if total > 0 else 0.0
    
    def _evaluate_loss_acc(self, dataloader, criterion):
        """Evaluate model loss and accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for (temporal, auxiliary), target in dataloader:
                temporal = temporal.to(self.device)
                auxiliary = auxiliary.to(self.device)
                target = target.to(self.device)
                output = self.model(temporal, auxiliary)
                loss = criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return total_loss / len(dataloader), correct / total if total > 0 else 0.0
    
    def _evaluate_with_metrics(self, dataloader):
        """Evaluate with confusion matrix and F1 score."""
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for (temporal, auxiliary), target in dataloader:
                temporal = temporal.to(self.device)
                auxiliary = auxiliary.to(self.device)
                target = target.to(self.device)
                output = self.model(temporal, auxiliary)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        acc = accuracy_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        return acc, cm, f1
    
    def _plot_training_curves(self, save_dir):
        """Plot and save training/validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax = axes[0]
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Accuracy curves
        ax = axes[1]
        ax.plot(epochs, [a*100 for a in self.train_accs], 'b-', label='Train Acc', linewidth=2)
        ax.plot(epochs, [a*100 for a in self.val_accs], 'r-', label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training and Validation Accuracy')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = f'{save_dir}/training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Training curves saved to {save_path}")
    
    def _plot_confusion_matrix(self, cm, save_dir):
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.shot_types, yticklabels=self.shot_types,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('TCN Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = f'{save_dir}/confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved to {save_path}")
    
    def test_batch(self, trajectories, labels):
        """Batch testing with comprehensive metrics."""
        print(f"\n{'='*70}")
        print("BATCH TESTING")
        print(f"{'='*70}")
        print(f"Testing on {len(trajectories)} trajectories...")
        
        dataset = TrajectoryDataset(trajectories, labels, self.label_encoder, self.scaler)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        acc, cm, f1 = self._evaluate_with_metrics(dataloader)
        
        print(f"\nTest Accuracy: {acc*100:.2f}%")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return acc, cm, f1
    
    def test_single(self, trajectory, true_label, save_path=None):
        """
        Test single trajectory with comprehensive 3D visualization.
        
        Args:
            trajectory: List of TrajectoryPoint
            true_label: True shot type label
            save_path: Optional path to save visualization
            
        Returns:
            prediction, confidence
        """
        self.model.eval()
        dataset = TrajectoryDataset([trajectory], [true_label], self.label_encoder, self.scaler)
        
        with torch.no_grad():
            (temporal, auxiliary), _ = dataset[0]
            temporal = temporal.unsqueeze(0).to(self.device)
            auxiliary = auxiliary.unsqueeze(0).to(self.device)
            output = self.model(temporal, auxiliary)
            probs = torch.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
            prediction = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # Visualize
        fig = plt.figure(figsize=(16, 6))
        
        # 3D trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        x = [p.x for p in trajectory]
        y = [p.y for p in trajectory]
        z = [p.z for p in trajectory]
        
        # Plot trajectory with color gradient
        for i in range(len(x)-1):
            ax1.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                    color=plt.cm.viridis(i/len(x)), linewidth=2)
        
        ax1.scatter([x[0]], [y[0]], [z[0]], c='green', s=200, marker='o', label='Start')
        ax1.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=200, marker='X', label='End')
        
        ax1.set_xlabel('X (m)', fontsize=10)
        ax1.set_ylabel('Y (m)', fontsize=10)
        ax1.set_zlabel('Z (m)', fontsize=10)
        ax1.set_title(f'3D Trajectory\nTrue: {true_label}\nPred: {prediction}', fontsize=11)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2D projections
        ax2 = fig.add_subplot(132)
        ax2.plot(x, z, 'b-', linewidth=2)
        ax2.scatter([x[0], x[-1]], [z[0], z[-1]], c=['green', 'red'], s=100)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Side View (X-Z Plane)')
        ax2.grid(alpha=0.3)
        
        # Confidence bar chart
        ax3 = fig.add_subplot(133)
        all_probs = probs[0].cpu().numpy()
        classes = [self.label_encoder.inverse_transform([i])[0] for i in range(len(all_probs))]
        colors = ['red' if c == prediction else 'lightblue' for c in classes]
        
        bars = ax3.bar(classes, all_probs, color=colors, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Confidence', fontsize=11)
        ax3.set_title('Class Probabilities', fontsize=11)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, all_probs)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add text box with info
        info_text = f"Prediction: {prediction}\nConfidence: {confidence:.2%}\nTrue Label: {true_label}\nCorrect: {prediction == true_label}"
        ax3.text(0.5, 0.97, info_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        print(f"\n{'='*70}")
        print(f"TCN Single Shot Prediction")
        print(f"{'='*70}")
        print(f"Prediction:  {prediction}")
        print(f"Confidence:  {confidence:.2%}")
        print(f"True Label:  {true_label}")
        print(f"Correct:     {prediction == true_label}")
        print(f"{'='*70}")
        
        return prediction, confidence
    
    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'config': self.config,
            'shot_types': self.shot_types,
            'train_history': {
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'val_losses': self.val_losses,
                'val_accs': self.val_accs
            }
        }, path)
    
    @staticmethod
    def load(path, device='cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']
        config.device = device
        
        trainer = TCNTrainTest(config)
        trainer.label_encoder = checkpoint['label_encoder']
        trainer.scaler = checkpoint['scaler']
        trainer.shot_types = checkpoint['shot_types']
        
        if 'train_history' in checkpoint:
            history = checkpoint['train_history']
            trainer.train_losses = history['train_losses']
            trainer.train_accs = history['train_accs']
            trainer.val_losses = history['val_losses']
            trainer.val_accs = history['val_accs']
        
        trainer.model = create_tcn_model(config, len(trainer.shot_types))
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.to(device)
        trainer.model.eval()
        
        return trainer


def load_data(data_path):
    """Load trajectory data from CSV file(s)."""
    data_files = glob.glob(data_path)
    if not data_files:
        raise FileNotFoundError(f"No data files found matching: {data_path}")
    
    print(f"Loading data from: {data_files[0]}")
    
    trajectories = defaultdict(list)
    current_shot = {}
    
    with open(data_files[0], 'r') as f:
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
    
    print(f"Loaded {len(all_traj)} trajectories")
    print(f"Shot types: {list(trajectories.keys())}")
    
    return all_traj, all_labels, trajectories


# CLI support
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python -m src.classifiers.tcn_train_test <mode> <data_path> [output_dir]")
        print("\nModes:")
        print("  train        - Train TCN model")
        print("  test_batch   - Batch testing on dataset")
        print("  test_single  - Single shot testing with visualization")
        print("\nExample:")
        print("  python -m src.classifiers.tcn_train_test train 'results/hockey_shots*.csv' tcn_models")
        sys.exit(1)
    
    mode = sys.argv[1]
    data_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'tcn_models'
    
    # Load data
    all_traj, all_labels, trajectories = load_data(data_path)
    
    if mode == 'train':
        trainer = TCNTrainTest()
        metrics = trainer.train(all_traj, all_labels, save_dir=output_dir)
        
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Train Accuracy: {metrics['train_acc']*100:.2f}%")
        print(f"Val Accuracy:   {metrics['val_acc']*100:.2f}%")
        print(f"Test Accuracy:  {metrics['test_acc']*100:.2f}%")
        print(f"Test F1 Score:  {metrics['f1_score']:.4f}")
        print(f"Best Epoch:     {metrics['best_epoch']+1}")
        print(f"{'='*70}")
        
        print(f"\n✅ Model saved to {output_dir}/best_tcn_model.pth")
    
    elif mode == 'test_batch':
        model_path = f'{output_dir}/best_tcn_model.pth'
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("Please train the model first using 'train' mode")
            sys.exit(1)
        
        trainer = TCNTrainTest.load(model_path)
        acc, cm, f1 = trainer.test_batch(all_traj, all_labels)
        
        print(f"\n{'='*70}")
        print("BATCH TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"F1 Score: {f1:.4f}")
        print(f"{'='*70}")
    
    elif mode == 'test_single':
        model_path = f'{output_dir}/best_tcn_model.pth'
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("Please train the model first using 'train' mode")
            sys.exit(1)
        
        trainer = TCNTrainTest.load(model_path)
        
        # Test first trajectory from each shot type
        for shot_type, traj_list in trajectories.items():
            test_traj = traj_list[0]
            save_path = f'{output_dir}/single_test_{shot_type}.png'
            trainer.test_single(test_traj, shot_type, save_path)
            print()
    
    else:
        print(f"❌ Unknown mode: {mode}")
        print("Valid modes: train, test_batch, test_single")
        sys.exit(1)

