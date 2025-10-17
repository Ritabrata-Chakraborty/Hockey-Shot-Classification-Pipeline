"""
Classical ML models (XGBoost, SVM, RF, KNN) training and testing.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
import os
from typing import List

from .feature_extraction import extract_trajectory_features
from ..physics.shot_logical import TrajectoryPoint, get_all_shot_types


class MLTrainer:
    def __init__(self, model_type='xgboost'):
        """
        Args:
            model_type: 'xgboost', 'svm', 'random_forest', 'knn'
        """
        self.model_type = model_type
        self.model = self._create_model()
        # Initialize scalers as dict (will be populated during training)
        self.scaler = {'temporal': StandardScaler(), 'auxiliary': StandardScaler()}
        self.label_encoder = LabelEncoder()
        self.shot_types = get_all_shot_types()
        self.label_encoder.fit(self.shot_types)
    
    def _create_model(self):
        """Create sklearn/xgboost model - optimized for 4000 samples."""
        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=300,          # More trees for larger dataset
                    max_depth=10,              # Deeper trees
                    learning_rate=0.05,        # Lower LR for better generalization
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,        # Regularization
                    gamma=0.1,                 # Minimum loss reduction
                    reg_alpha=0.1,             # L1 regularization
                    reg_lambda=1.0,            # L2 regularization
                    random_state=42,
                    eval_metric='mlogloss',
                    n_jobs=-1)
            except ImportError:
                print("Warning: XGBoost not installed. Install with: pip install xgboost")
                raise
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf', 
                C=100.0,                       # Higher C for complex patterns
                gamma='scale',                 # Auto-scale gamma
                probability=True,
                class_weight='balanced',       # Handle any imbalance
                random_state=42,
                cache_size=1000)               # Larger cache for speed
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=300,              # More trees
                max_depth=20,                  # Deeper trees
                min_samples_split=5,
                min_samples_leaf=2,            # Allow finer splits
                max_features='sqrt',           # Feature sampling
                bootstrap=True,
                class_weight='balanced',       # Handle imbalance
                random_state=42,
                n_jobs=-1)
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=15,                # More neighbors for 4K samples
                weights='distance',            # Distance weighting
                metric='euclidean',
                algorithm='auto',              # Auto-select best algorithm
                n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, trajectories, labels, test_val_size=0.2):
        """
        Train model on aggregated features.
        
        Args:
            trajectories: List of trajectory lists
            labels: List of labels
            test_val_size: Combined validation + test size (0.2 = 10% val + 10% test)
        
        Returns: dict with train_acc, val_acc, test_acc, confusion_matrix
        """
        print(f"Training {self.model_type.upper()} on {len(trajectories)} trajectories...")
        print("Extracting features...")
        
        # Extract features from all trajectories
        X = np.array([extract_trajectory_features(t) for t in trajectories])
        y = self.label_encoder.transform(labels)
        
        print(f"Feature shape: {X.shape}")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("⚠️  Warning: NaN or Inf values detected in features, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data: 80% train, 10% val, 10% test (consistent with TCN)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        # Split features: temporal (600) + auxiliary (52) for consistent normalization with TCN
        print("\nNormalizing features (separate temporal + auxiliary)...")
        X_train_temporal = X_train[:, :600]
        X_train_auxiliary = X_train[:, 600:]
        X_val_temporal = X_val[:, :600]
        X_val_auxiliary = X_val[:, 600:]
        X_test_temporal = X_test[:, :600]
        X_test_auxiliary = X_test[:, 600:]
        
        # Separate normalization (consistent with TCN)
        scaler_temporal = StandardScaler()
        scaler_auxiliary = StandardScaler()
        
        X_train_temporal = scaler_temporal.fit_transform(X_train_temporal)
        X_train_auxiliary = scaler_auxiliary.fit_transform(X_train_auxiliary)
        
        X_val_temporal = scaler_temporal.transform(X_val_temporal)
        X_val_auxiliary = scaler_auxiliary.transform(X_val_auxiliary)
        
        X_test_temporal = scaler_temporal.transform(X_test_temporal)
        X_test_auxiliary = scaler_auxiliary.transform(X_test_auxiliary)
        
        # Recombine
        X_train = np.hstack([X_train_temporal, X_train_auxiliary])
        X_val = np.hstack([X_val_temporal, X_val_auxiliary])
        X_test = np.hstack([X_test_temporal, X_test_auxiliary])
        
        # Store both scalers for later use
        self.scaler = {'temporal': scaler_temporal, 'auxiliary': scaler_auxiliary}
        
        print("Training model...")
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        test_acc = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'confusion_matrix': cm,
            'f1_score': f1
        }
    
    def test_batch(self, trajectories, labels):
        """Batch testing."""
        X = np.array([extract_trajectory_features(t) for t in trajectories])
        
        # Apply same normalization strategy
        X_temporal = X[:, :600]
        X_auxiliary = X[:, 600:]
        X_temporal_scaled = self.scaler['temporal'].transform(X_temporal)
        X_auxiliary_scaled = self.scaler['auxiliary'].transform(X_auxiliary)
        X_scaled = np.hstack([X_temporal_scaled, X_auxiliary_scaled])
        
        y_true = self.label_encoder.transform(labels)
        y_pred = self.model.predict(X_scaled)
        
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return acc, cm
    
    def test_single(self, trajectory, true_label, save_path=None):
        """
        Test single trajectory with visualization.
        Shows trajectory + feature importance/confidence + prediction.
        """
        features = extract_trajectory_features(trajectory)
        
        # Apply same normalization strategy
        X_temporal = features[:600].reshape(1, -1)
        X_auxiliary = features[600:].reshape(1, -1)
        X_temporal_scaled = self.scaler['temporal'].transform(X_temporal)
        X_auxiliary_scaled = self.scaler['auxiliary'].transform(X_auxiliary)
        X = np.hstack([X_temporal_scaled, X_auxiliary_scaled])
        
        pred_idx = self.model.predict(X)[0]
        prediction = self.label_encoder.inverse_transform([pred_idx])[0]
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = proba[pred_idx]
            all_probs = proba
        else:
            confidence = 1.0
            all_probs = None
        
        # Visualize - 3 panel layout matching TCN format
        fig = plt.figure(figsize=(16, 6))
        
        # Panel 1: 3D trajectory
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
        
        # Panel 2: 2D projection (X-Z plane)
        ax2 = fig.add_subplot(132)
        ax2.plot(x, z, 'b-', linewidth=2)
        ax2.scatter([x[0], x[-1]], [z[0], z[-1]], c=['green', 'red'], s=100)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Side View (X-Z Plane)')
        ax2.grid(alpha=0.3)
        
        # Panel 3: Confidence bar chart
        ax3 = fig.add_subplot(133)
        
        if all_probs is not None:
            # Show class probabilities
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
            info_text = f"Model: {self.model_type.upper()}\nPrediction: {prediction}\nConfidence: {confidence:.2%}\nTrue Label: {true_label}\nCorrect: {prediction == true_label}"
            ax3.text(0.5, 0.97, info_text, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # For models without predict_proba, show model info
            ax3.text(0.5, 0.5, f"Model: {self.model_type.upper()}\nPredicted: {prediction}\nTrue: {true_label}\nCorrect: {prediction == true_label}", 
                    transform=ax3.transAxes, fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax3.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"{self.model_type.upper()} Prediction: {prediction} (confidence: {confidence:.2%})")
        print(f"True Label: {true_label}")
        print(f"Correct: {prediction == true_label}")
        
        return prediction, confidence
    
    def save(self, path):
        """Save model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
    
    @staticmethod
    def load(path):
        """Load model."""
        return joblib.load(path)


# CLI support
if __name__ == '__main__':
    import sys, csv, glob
    from collections import defaultdict
    
    if len(sys.argv) < 4:
        print("Usage: python -m src.classifiers.ml_trainer <model_type> <mode> <data_path> [model_path]")
        print("Models: xgboost, svm, random_forest, knn")
        print("Modes: train, test_batch, test_single")
        sys.exit(1)
    
    model_type = sys.argv[1]
    mode = sys.argv[2]
    data_path = sys.argv[3]
    
    # Load data
    data_files = glob.glob(data_path)
    if not data_files:
        print('❌ No data files found')
        sys.exit(1)
    
    trajectories, current_shot = defaultdict(list), {}
    with open(data_files[0], 'r') as f:
        for row in csv.DictReader(f):
            key = (row['shot_type'], int(row['shot_id']))
            if key not in current_shot:
                current_shot[key] = []
            current_shot[key].append(TrajectoryPoint(
                float(row['time']), float(row['x']), float(row['y']), float(row['z'])))
    
    for (shot_type, _), points in current_shot.items():
        trajectories[shot_type].append(sorted(points, key=lambda p: p.time))
    
    all_traj, all_labels = [], []
    for shot_type, traj_list in trajectories.items():
        all_traj.extend(traj_list)
        all_labels.extend([shot_type] * len(traj_list))
    
    if mode == 'train':
        trainer = MLTrainer(model_type)
        metrics = trainer.train(all_traj, all_labels)
        print(f"\nTrain Acc: {metrics['train_acc']*100:.2f}%")
        print(f"Val Acc:   {metrics['val_acc']*100:.2f}%")
        print(f"Test Acc:  {metrics['test_acc']*100:.2f}%")
        
        if len(sys.argv) > 4:
            trainer.save(sys.argv[4])
            print(f"Model saved to {sys.argv[4]}")
    
    elif mode == 'test_single':
        if len(sys.argv) < 5:
            print("Usage: python -m src.classifiers.ml_trainer <model_type> test_single <data_path> <model_path>")
            sys.exit(1)
        
        model_path = sys.argv[4]
        trainer = MLTrainer.load(model_path)
        
        # Test first trajectory
        shot_type = list(trajectories.keys())[0]
        test_traj = trajectories[shot_type][0]
        save_path = model_path.replace('.pkl', '_single_test.png')
        trainer.test_single(test_traj, shot_type, save_path)
        print(f"Visualization saved to {save_path}")

