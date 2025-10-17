"""
Unified Shot Control System
Combines adaptive control and real-time correction for hockey shot classification.

Features:
- Real-time shot type detection using any classifier (TCN, XGBoost, SVM, RF, KNN)
- Deviation analysis from intended shot type
- PID-based automatic correction
- Environmental disturbance handling (wind, humidity, faults)
- Iterative refinement until target achieved
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime
import os

from ..classifiers.tcn_train_test import TCNTrainTest
from ..classifiers.ml_trainer import MLTrainer
from ..physics.shot_logical import TrajectoryPoint, get_all_shot_types
from ..physics.effects_simulation import (
    AdvancedPhysicsEngine, 
    EnvironmentalConditions, 
    SpinParameters,
    EffectsPresets
)
from ..classifiers.feature_extraction import extract_all_features


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DeviationAnalysis:
    """Analysis of deviation from intended shot"""
    intended_type: str
    predicted_type: str
    prediction_confidence: float
    deviation_score: float  # 0-1, how far from intended
    velocity_error: float
    angle_error: float
    trajectory_error: float  # RMS error in trajectory
    spin_error: float
    class_probabilities: Dict[str, float]


@dataclass
class CorrectionAction:
    """Corrective actions to achieve intended shot"""
    velocity_adjustment: float  # m/s
    launch_angle_adjustment: float  # degrees
    lateral_angle_adjustment: float  # degrees
    spin_rate_adjustment: float  # rad/s
    spin_axis_adjustment: Tuple[float, float, float]
    confidence: float


@dataclass
class CorrectionResult:
    """Result of applying correction"""
    before_deviation: DeviationAnalysis
    after_correction: DeviationAnalysis
    correction_applied: CorrectionAction
    success: bool
    improvement_percent: float
    iterations_needed: int
    trajectory_before: List[TrajectoryPoint]
    trajectory_after: List[TrajectoryPoint]


# ============================================================================
# Main Control System
# ============================================================================

class ShotControlSystem:
    """
    Unified shot control system for real-time detection and correction.
    
    Workflow:
    1. Detect actual shot type using classifier
    2. Analyze deviation from intended shot
    3. Calculate PID corrections
    4. Apply corrections iteratively
    5. Verify target achieved
    """
    
    def __init__(self, classifier_path: str, classifier_type: str = 'tcn',
                 max_iterations: int = 10, tolerance: float = 0.15):
        """
        Args:
            classifier_path: Path to trained classifier model
            classifier_type: 'tcn', 'xgboost', 'svm', 'random_forest', 'knn'
            max_iterations: Max correction iterations
            tolerance: Acceptable deviation (0-1, 0=perfect)
        """
        self.classifier_type = classifier_type
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Load classifier
        print(f"Loading {classifier_type.upper()} classifier...")
        if classifier_type == 'tcn':
            self.classifier = TCNTrainTest.load(classifier_path, device='cpu')
        else:
            self.classifier = MLTrainer.load(classifier_path)
        
        self.shot_types = get_all_shot_types()
        
        # Reference parameters for each shot type
        self.reference_params = {
            'hit': {'velocity': 35.0, 'launch_angle': 12.0, 'spin_rate': 15.0},
            'slap': {'velocity': 45.0, 'launch_angle': 15.0, 'spin_rate': 25.0},
            'push_flick': {'velocity': 28.0, 'launch_angle': 18.0, 'spin_rate': 10.0},
            'drag_flick': {'velocity': 32.0, 'launch_angle': 22.0, 'spin_rate': 35.0}
        }
        
        # PID controller gains (more aggressive for better correction)
        self.gains = {
            'velocity': {'kp': 1.0, 'ki': 0.2, 'kd': 0.4},
            'angle': {'kp': 0.8, 'ki': 0.1, 'kd': 0.3},
            'spin': {'kp': 0.6, 'ki': 0.1, 'kd': 0.2}
        }
        
        # PID state variables
        self.reset_pid()
        
        # History
        self.correction_history = []
        
        print("✅ Control system initialized")
    
    def reset_pid(self):
        """Reset PID controller state"""
        self.velocity_integral = 0.0
        self.angle_integral = 0.0
        self.spin_integral = 0.0
        self.prev_velocity_error = 0.0
        self.prev_angle_error = 0.0
        self.prev_spin_error = 0.0
    
    def predict_shot_type(self, trajectory: List[TrajectoryPoint]) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict shot type from trajectory using loaded classifier.
        
        Returns:
            (predicted_type, confidence, class_probabilities)
        """
        if self.classifier_type == 'tcn':
            # TCN prediction
            features = extract_all_features(trajectory, aligned=True)
            temporal = features[:600].reshape(3, 200)
            auxiliary = features[600:]
            
            temporal_tensor = torch.FloatTensor(temporal).unsqueeze(0)
            auxiliary_tensor = torch.FloatTensor(auxiliary).unsqueeze(0)
            
            self.classifier.model.eval()
            with torch.no_grad():
                output = self.classifier.model(temporal_tensor, auxiliary_tensor)
                probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
            
            pred_idx = probabilities.argmax()
            predicted_type = self.shot_types[pred_idx]
            confidence = probabilities[pred_idx]
            
            probs_dict = {shot_type: float(probabilities[i]) 
                         for i, shot_type in enumerate(self.shot_types)}
        else:
            # Classical ML prediction
            features = extract_all_features(trajectory, aligned=True)
            features_temporal = features[:600].reshape(1, -1)
            features_auxiliary = features[600:].reshape(1, -1)
            
            features_temporal_scaled = self.classifier.scaler['temporal'].transform(features_temporal)
            features_auxiliary_scaled = self.classifier.scaler['auxiliary'].transform(features_auxiliary)
            features_normalized = np.hstack([features_temporal_scaled, features_auxiliary_scaled])
            
            pred_idx = self.classifier.model.predict(features_normalized)[0]
            predicted_type = self.classifier.label_encoder.inverse_transform([pred_idx])[0]
            
            if hasattr(self.classifier.model, 'predict_proba'):
                probabilities = self.classifier.model.predict_proba(features_normalized)[0]
                confidence = probabilities[pred_idx]
                probs_dict = {shot_type: float(probabilities[i]) 
                             for i, shot_type in enumerate(self.shot_types)}
            else:
                confidence = 1.0
                probs_dict = {predicted_type: 1.0}
        
        return predicted_type, confidence, probs_dict
    
    def analyze_deviation(self, intended_type: str, trajectory: List[TrajectoryPoint],
                         actual_params: Dict[str, float]) -> DeviationAnalysis:
        """Analyze how much the shot deviates from intended type."""
        predicted_type, confidence, probs = self.predict_shot_type(trajectory)
        
        ref_params = self.reference_params[intended_type]
        
        # Calculate parameter errors
        velocity_error = actual_params['velocity'] - ref_params['velocity']
        angle_error = actual_params.get('launch_angle', 15.0) - ref_params['launch_angle']
        spin_error = actual_params.get('spin_rate', 0.0) - ref_params['spin_rate']
        
        # Calculate trajectory error
        max_range = max(p.x for p in trajectory)
        max_height = max(p.z for p in trajectory)
        
        ref_ranges = {'hit': 130, 'slap': 15, 'push_flick': 60, 'drag_flick': 45}
        ref_heights = {'hit': 8, 'slap': 1.5, 'push_flick': 5.5, 'drag_flick': 5}
        
        range_error = abs(max_range - ref_ranges[intended_type]) / ref_ranges[intended_type]
        height_error = abs(max_height - ref_heights[intended_type]) / ref_heights[intended_type]
        trajectory_error = np.sqrt(range_error**2 + height_error**2)
        
        # Overall deviation (1 - probability of intended class)
        deviation_score = 1.0 - probs.get(intended_type, 0.0)
        
        return DeviationAnalysis(
            intended_type=intended_type,
            predicted_type=predicted_type,
            prediction_confidence=confidence,
            deviation_score=deviation_score,
            velocity_error=velocity_error,
            angle_error=angle_error,
            trajectory_error=trajectory_error,
            spin_error=spin_error,
            class_probabilities=probs
        )
    
    def calculate_correction(self, deviation: DeviationAnalysis, dt: float = 1.0) -> CorrectionAction:
        """Calculate PID-based correction to minimize deviation."""
        # Update integrals
        self.velocity_integral += deviation.velocity_error * dt
        self.angle_integral += deviation.angle_error * dt
        self.spin_integral += deviation.spin_error * dt
        
        # Calculate derivatives
        velocity_derivative = (deviation.velocity_error - self.prev_velocity_error) / dt
        angle_derivative = (deviation.angle_error - self.prev_angle_error) / dt
        spin_derivative = (deviation.spin_error - self.prev_spin_error) / dt
        
        # PID corrections
        velocity_correction = -(
            self.gains['velocity']['kp'] * deviation.velocity_error +
            self.gains['velocity']['ki'] * self.velocity_integral +
            self.gains['velocity']['kd'] * velocity_derivative
        )
        
        angle_correction = -(
            self.gains['angle']['kp'] * deviation.angle_error +
            self.gains['angle']['ki'] * self.angle_integral +
            self.gains['angle']['kd'] * angle_derivative
        )
        
        spin_correction = -(
            self.gains['spin']['kp'] * deviation.spin_error +
            self.gains['spin']['ki'] * self.spin_integral +
            self.gains['spin']['kd'] * spin_derivative
        )
        
        # Update previous errors
        self.prev_velocity_error = deviation.velocity_error
        self.prev_angle_error = deviation.angle_error
        self.prev_spin_error = deviation.spin_error
        
        # Lateral correction (proportional to trajectory error)
        lateral_correction = -deviation.trajectory_error * 5.0
        
        # Confidence
        confidence = max(0.0, 1.0 - deviation.deviation_score)
        
        return CorrectionAction(
            velocity_adjustment=velocity_correction,
            launch_angle_adjustment=angle_correction,
            lateral_angle_adjustment=lateral_correction,
            spin_rate_adjustment=spin_correction,
            spin_axis_adjustment=(0.0, 0.0, 1.0),
            confidence=confidence
        )
    
    def simulate_trajectory(self, params: Dict[str, float], 
                          conditions: EnvironmentalConditions) -> List[TrajectoryPoint]:
        """Simulate trajectory with given parameters and conditions."""
        spin_params = SpinParameters(
            spin_rate=params.get('spin_rate', 0.0),
            spin_axis=(0, 0, 1)
        )
        
        engine = AdvancedPhysicsEngine(conditions)
        return engine.calculate_advanced_trajectory(
            initial_velocity=params['velocity'],
            launch_angle=params.get('launch_angle', 15.0),
            launch_height=0.5,
            lateral_angle=params.get('lateral_angle', 0.0),
            spin_params=spin_params,
            time_step=0.01,
            max_time=5.0
        )
    
    def correct_shot(self, intended_type: str, 
                    actual_params: Dict[str, float],
                    disturbance: EnvironmentalConditions = None) -> CorrectionResult:
        """
        Main correction loop: iteratively apply corrections until target achieved.
        
        Args:
            intended_type: Desired shot type
            actual_params: Current parameters (with disturbances/faults)
            disturbance: Environmental conditions
        
        Returns:
            CorrectionResult with full analysis
        """
        if disturbance is None:
            disturbance = EffectsPresets.calm_conditions()
        
        self.reset_pid()
        
        # Simulate initial trajectory
        trajectory_before = self.simulate_trajectory(actual_params, disturbance)
        deviation_before = self.analyze_deviation(intended_type, trajectory_before, actual_params)
        
        print(f"\n{'='*70}")
        print(f"SHOT CORRECTION SYSTEM - {self.classifier_type.upper()}")
        print(f"{'='*70}")
        print(f"Intended: {intended_type}")
        print(f"Detected: {deviation_before.predicted_type} (confidence: {deviation_before.prediction_confidence:.1%})")
        print(f"Deviation: {deviation_before.deviation_score:.1%}")
        print(f"  Velocity error: {deviation_before.velocity_error:+.1f} m/s")
        print(f"  Angle error: {deviation_before.angle_error:+.1f}°")
        print(f"  Spin error: {deviation_before.spin_error:+.1f} rad/s")
        print()
        
        # Iterative correction
        current_params = actual_params.copy()
        iteration = 0
        success = False
        
        while iteration < self.max_iterations:
            iteration += 1
            
            trajectory_current = self.simulate_trajectory(current_params, disturbance)
            deviation_current = self.analyze_deviation(intended_type, trajectory_current, current_params)
            
            print(f"Iteration {iteration}: {deviation_current.predicted_type} "
                  f"(confidence: {deviation_current.prediction_confidence:.1%}, "
                  f"deviation: {deviation_current.deviation_score:.1%})")
            
            if deviation_current.deviation_score < self.tolerance:
                print(f"  ✅ Target achieved!")
                success = True
                trajectory_after = trajectory_current
                deviation_after = deviation_current
                break
            
            correction = self.calculate_correction(deviation_current, dt=1.0)
            
            print(f"  Correction: Δv={correction.velocity_adjustment:+.1f} m/s, "
                  f"Δα={correction.launch_angle_adjustment:+.1f}°")
            
            current_params['velocity'] += correction.velocity_adjustment
            current_params['launch_angle'] = current_params.get('launch_angle', 15.0) + correction.launch_angle_adjustment
            current_params['spin_rate'] = current_params.get('spin_rate', 0.0) + correction.spin_rate_adjustment
        
        if not success:
            print(f"\n⚠️  Max iterations reached. Final deviation: {deviation_current.deviation_score:.1%}")
            trajectory_after = trajectory_current
            deviation_after = deviation_current
        
        # Calculate improvement (avoid division by zero)
        if deviation_before.deviation_score > 0:
            improvement = ((deviation_before.deviation_score - deviation_after.deviation_score) / 
                          deviation_before.deviation_score * 100)
        else:
            improvement = 0.0
        
        print(f"\n{'='*70}")
        print(f"RESULTS: {improvement:+.1f}% improvement in {iteration} iterations")
        print(f"{'='*70}\n")
        
        # Create default correction if target achieved immediately
        if 'correction' not in locals():
            correction = CorrectionAction(
                velocity_adjustment=0.0,
                launch_angle_adjustment=0.0,
                lateral_angle_adjustment=0.0,
                spin_rate_adjustment=0.0,
                spin_axis_adjustment=(0.0, 0.0, 0.0),
                confidence=1.0
            )
        
        result = CorrectionResult(
            before_deviation=deviation_before,
            after_correction=deviation_after,
            correction_applied=correction,
            success=success,
            improvement_percent=improvement,
            iterations_needed=iteration,
            trajectory_before=trajectory_before,
            trajectory_after=trajectory_after
        )
        
        self.correction_history.append(result)
        return result
    
    def visualize_correction(self, result: CorrectionResult, save_path: str = None):
        """Visualize correction with before/after trajectories."""
        fig = plt.figure(figsize=(16, 10))
        
        # 3D trajectory
        ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d')
        
        x_before = [p.x for p in result.trajectory_before]
        y_before = [p.y for p in result.trajectory_before]
        z_before = [p.z for p in result.trajectory_before]
        
        x_after = [p.x for p in result.trajectory_after]
        y_after = [p.y for p in result.trajectory_after]
        z_after = [p.z for p in result.trajectory_after]
        
        ax1.plot(x_before, y_before, z_before, 'r-', linewidth=2, alpha=0.7, label='Before')
        ax1.plot(x_after, y_after, z_after, 'g-', linewidth=2, alpha=0.7, label='After')
        ax1.set_xlabel('Range (m)')
        ax1.set_ylabel('Lateral (m)')
        ax1.set_zlabel('Height (m)')
        ax1.set_title(f'Shot Correction: {result.before_deviation.intended_type}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error bars
        ax2 = fig.add_subplot(2, 3, 2)
        errors = ['Velocity', 'Angle', 'Spin', 'Trajectory']
        before = [abs(result.before_deviation.velocity_error),
                 abs(result.before_deviation.angle_error),
                 abs(result.before_deviation.spin_error),
                 result.before_deviation.trajectory_error]
        after = [abs(result.after_correction.velocity_error),
                abs(result.after_correction.angle_error),
                abs(result.after_correction.spin_error),
                result.after_correction.trajectory_error]
        
        x = np.arange(len(errors))
        width = 0.35
        ax2.bar(x - width/2, before, width, label='Before', color='red', alpha=0.7)
        ax2.bar(x + width/2, after, width, label='After', color='green', alpha=0.7)
        ax2.set_ylabel('Error')
        ax2.set_title('Error Reduction')
        ax2.set_xticks(x)
        ax2.set_xticklabels(errors)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Probabilities
        ax3 = fig.add_subplot(2, 3, 3)
        shot_types = list(result.before_deviation.class_probabilities.keys())
        before_probs = list(result.before_deviation.class_probabilities.values())
        after_probs = list(result.after_correction.class_probabilities.values())
        
        x = np.arange(len(shot_types))
        width = 0.35
        ax3.bar(x - width/2, before_probs, width, label='Before', alpha=0.7)
        ax3.bar(x + width/2, after_probs, width, label='After', alpha=0.7)
        
        intended_idx = shot_types.index(result.before_deviation.intended_type)
        ax3.axvline(intended_idx, color='red', linestyle='--', alpha=0.5, label='Target')
        
        ax3.set_ylabel('Probability')
        ax3.set_title('Classification Confidence')
        ax3.set_xticks(x)
        ax3.set_xticklabels(shot_types, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Summary
        ax4 = fig.add_subplot(2, 3, 5)
        ax4.axis('off')
        summary = f"""
SUMMARY

Target: {result.before_deviation.intended_type}

Before:
  Detected: {result.before_deviation.predicted_type}
  Confidence: {result.before_deviation.prediction_confidence:.1%}
  Deviation: {result.before_deviation.deviation_score:.1%}

After:
  Detected: {result.after_correction.predicted_type}
  Confidence: {result.after_correction.prediction_confidence:.1%}
  Deviation: {result.after_correction.deviation_score:.1%}

Result:
  Iterations: {result.iterations_needed}
  Improvement: {result.improvement_percent:.1f}%
  Success: {'✅' if result.success else '❌'}
        """
        ax4.text(0.1, 0.5, summary, transform=ax4.transAxes, 
                fontsize=10, family='monospace', verticalalignment='center')
        
        # Side view
        ax5 = fig.add_subplot(2, 3, 6)
        ax5.plot(x_before, z_before, 'r-', linewidth=2, alpha=0.7, label='Before')
        ax5.plot(x_after, z_after, 'g-', linewidth=2, alpha=0.7, label='After')
        ax5.set_xlabel('Range (m)')
        ax5.set_ylabel('Height (m)')
        ax5.set_title('Side View')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        plt.close()


# ============================================================================
# Factory Function
# ============================================================================

def create_shot_control_system(classifier_path: str, classifier_type: str = 'tcn') -> ShotControlSystem:
    """
    Create a shot control system.
    
    Args:
        classifier_path: Path to trained model
        classifier_type: 'tcn', 'xgboost', 'svm', 'random_forest', 'knn'
    
    Returns:
        ShotControlSystem instance
    """
    return ShotControlSystem(classifier_path, classifier_type)

