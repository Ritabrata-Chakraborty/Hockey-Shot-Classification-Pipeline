#!/usr/bin/env python3
"""
CLI wrapper for shot control system demonstration.
"""

import sys
import argparse
import os
from .shot_control import ShotControlSystem
from ..physics.effects_simulation import EffectsPresets

def main():
    parser = argparse.ArgumentParser(description='Run shot correction system demo')
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('model_type', type=str, 
                       choices=['tcn', 'xgboost', 'svm', 'random_forest', 'knn'],
                       help='Type of model')
    parser.add_argument('--shot-type', type=str, default='hit',
                       choices=['hit', 'slap', 'push_flick', 'drag_flick'],
                       help='Intended shot type to test')
    parser.add_argument('--output', type=str, default='models/control_results',
                       help='Output directory for results')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum correction iterations')
    parser.add_argument('--tolerance', type=float, default=0.15,
                       help='Acceptable deviation (0-1)')
    
    args = parser.parse_args()
    
    print(f"Loading {args.model_type.upper()} model from: {args.model_path}")
    
    # Initialize control system
    system = ShotControlSystem(
        classifier_path=args.model_path,
        classifier_type=args.model_type,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance
    )
    
    # Run demonstration scenarios
    print(f"\n{'='*70}")
    print(f"SHOT CORRECTION SYSTEM DEMO")
    print(f"{'='*70}")
    print(f"Model: {args.model_type.upper()}")
    print(f"Intended Shot: {args.shot_type.upper()}")
    print()
    
    # Test scenarios optimized for demo
    # These parameters are deliberately wrong to trigger misclassification
    # The PID controller will correct them toward the intended shot type
    # 
    # Note: Parameters chosen to maximize chance of initial misclassification
    # while still being correctable through PID iterations
    test_scenarios = {
        'hit': {
            'velocity': 42.0,      # Too fast - closer to slap (45)
            'launch_angle': 14.5,  # Between hit (12) and slap (15)
            'lateral_angle': 0.8,
            'spin_rate': 20.0,     # Between hit (15) and slap (25)
            'conditions': 'calm'
        },
        'slap': {
            'velocity': 33.0,      # Much too slow (ref: 45)
            'launch_angle': 11.0,  # Much too low (ref: 15)
            'lateral_angle': 0.2,
            'spin_rate': 12.0,     # Much too low (ref: 25)
            'conditions': 'calm'
        },
        'push_flick': {
            'velocity': 30.0,      # Slightly high
            'launch_angle': 16.0,  # Between push_flick (18) and slap (15)
            'lateral_angle': 0.3,
            'spin_rate': 13.0,     # Between push_flick (10) and hit (15)
            'conditions': 'calm'
        },
        'drag_flick': {
            'velocity': 29.5,      # Closer to push_flick (28)
            'launch_angle': 19.0,  # Between push_flick (18) and drag_flick (22)
            'lateral_angle': 0.8,
            'spin_rate': 15.0,     # Much too low (ref: 35)
            'conditions': 'calm'
        }
    }
    
    # Get scenario for requested shot type
    scenario = test_scenarios.get(args.shot_type, test_scenarios['hit'])
    
    actual_params = {
        'velocity': scenario['velocity'],
        'launch_angle': scenario['launch_angle'],
        'lateral_angle': scenario['lateral_angle'],
        'spin_rate': scenario['spin_rate']
    }
    
    # Select environmental conditions
    conditions_map = {
        'calm': EffectsPresets.calm_conditions(),
        'windy': EffectsPresets.windy_conditions(),
        'humid': EffectsPresets.humid_conditions()
    }
    disturbance = conditions_map.get(scenario['conditions'], EffectsPresets.calm_conditions())
    
    result = system.correct_shot(
        intended_type=args.shot_type,
        actual_params=actual_params,
        disturbance=disturbance
    )
    
    # Generate visualization
    os.makedirs(args.output, exist_ok=True)
    viz_path = f'{args.output}/correction_{args.shot_type}.png'
    system.visualize_correction(result, save_path=viz_path)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Intended Type:      {args.shot_type.upper()}")
    print(f"Initial Prediction: {result.before_deviation.predicted_type.upper()}")
    print(f"Final Prediction:   {result.after_correction.predicted_type.upper()}")
    print(f"Success:            {result.success}")
    print(f"Iterations:         {result.iterations_needed}")
    print(f"Improvement:        {result.improvement_percent:.1f}%")
    print(f"Initial Confidence: {result.before_deviation.prediction_confidence:.1%}")
    print(f"Final Confidence:   {result.after_correction.prediction_confidence:.1%}")
    print(f"Deviation:          {result.before_deviation.deviation_score:.3f} -> {result.after_correction.deviation_score:.3f}")
    print(f"\nVisualization saved to: {viz_path}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()

