import csv, random, argparse, os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from .shot_logical import (
    hit_drive_shot, slap_shot, push_flick_shot, drag_flick_shot,
    calculate_trajectory, get_parameter_ranges, get_trajectory_settings,
    shot_type_from_string, get_all_shot_types, ShotResult, TrajectoryPoint
)


class ShotGenerator:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.parameter_ranges = get_parameter_ranges()
        self.trajectory_settings = get_trajectory_settings()
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_random_parameters(self, shot_type: str) -> Dict:
        shot_type = shot_type_from_string(shot_type)
        ranges = self.parameter_ranges[shot_type]
        
        params = {}
        for param_name, (min_val, max_val) in ranges.items():
            # Add more realistic parameter distributions
            if param_name in ['F', 'force']:  # Force parameters
                # Use normal distribution for force (most shots are medium force)
                mean_force = (min_val + max_val) / 2
                std_force = (max_val - min_val) / 6  # 99.7% within range
                params[param_name] = max(min_val, min(max_val, random.gauss(mean_force, std_force)))
            
            elif param_name in ['t', 'time', 'dt']:  # Time parameters
                # Slightly favor shorter contact times (more common)
                params[param_name] = random.triangular(min_val, max_val, min_val + (max_val - min_val) * 0.3)
            
            elif param_name in ['alpha', 'angle']:  # Angle parameters
                # Normal distribution around middle values
                mean_angle = (min_val + max_val) / 2
                std_angle = (max_val - min_val) / 4
                params[param_name] = max(min_val, min(max_val, random.gauss(mean_angle, std_angle)))
            
            elif param_name in ['L', 'length', 'distance']:  # Length/distance parameters
                # Uniform distribution but with slight bias toward middle-high values
                params[param_name] = random.triangular(min_val, max_val, min_val + (max_val - min_val) * 0.6)
            
            else:  # Default uniform distribution
                params[param_name] = random.uniform(min_val, max_val)
        
        return params
    
    def calculate_shot_physics(self, shot_type: str, params: Dict) -> ShotResult:
        shot_type = shot_type_from_string(shot_type)
        shots = {"hit": hit_drive_shot, "slap": slap_shot, "push_flick": push_flick_shot, "drag_flick": drag_flick_shot}
        return shots[shot_type](**params)
    
    def generate_single_shot(self, shot_type: str, shot_id: int = 0, 
                           custom_params: Optional[Dict] = None) -> Tuple[ShotResult, List[TrajectoryPoint]]:
        params = self.generate_random_parameters(shot_type) if custom_params is None else custom_params.copy()
        shot_result = self.calculate_shot_physics(shot_type, params)
        
        from .shot_logical import apply_stick_flex_enhancement
        from .effects_simulation import (AdvancedPhysicsEngine, EnvironmentalConditions, 
                                        SpinParameters, EffectsPresets)
        
        # Apply biomechanical enhancements
        stick_flex = random.uniform(0.10, 0.25)
        contact_time = random.uniform(0.005, 0.012)
        player_strength = max(0.7, min(1.3, random.gauss(1.0, 0.15)))
        shot_result.velocity = apply_stick_flex_enhancement(shot_result.velocity, stick_flex, contact_time, player_strength)
        
        # Shot-type-specific launch angles (consistent with advanced physics)
        launch_angles = {'hit': 12.0, 'slap': 15.0, 'push_flick': 18.0, 'drag_flick': 22.0}
        base_angle = launch_angles.get(shot_type, 15.0)
        launch_angle = random.uniform(base_angle - 3, base_angle + 3)
        
        launch_height = random.uniform(0.3, 0.7)
        lateral_angle = random.uniform(-20, 20)
        
        # Random environmental conditions for variety (60% calm, 20% windy, 20% humid)
        rand_val = random.random()
        if rand_val < 0.60:
            conditions = EffectsPresets.calm_conditions()
        elif rand_val < 0.80:
            conditions = EffectsPresets.windy_conditions()
        else:
            conditions = EffectsPresets.humid_conditions()
        
        # Shot-type-specific spin parameters (from advanced physics)
        spin_configs = {
            'hit': SpinParameters(15.0, (0.2, 0.1, 0.97)),
            'slap': SpinParameters(25.0, (0.1, 0.0, 0.99)),
            'push_flick': SpinParameters(10.0, (0.0, 0.3, 0.95)),
            'drag_flick': SpinParameters(35.0, (0.3, 0.2, 0.93))
        }
        spin_params = spin_configs.get(shot_type, SpinParameters())
        
        # Use advanced physics engine for realistic trajectories
        engine = AdvancedPhysicsEngine(conditions)
        trajectory = engine.calculate_advanced_trajectory(
            shot_result.velocity,
            launch_angle,
            launch_height,
            lateral_angle,
            spin_params,
            self.trajectory_settings["time_step"],
            self.trajectory_settings["max_time"]
        )
        
        return shot_result, trajectory
    
    def generate_multiple_shots(self, shot_type: str, num_shots: int, 
                              progress_callback: Optional[callable] = None) -> List[Tuple[ShotResult, List[TrajectoryPoint]]]:
        """
        Generate multiple shots of the same type.
        
        Args:
            shot_type: Type of shot
            num_shots: Number of shots to generate
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of (ShotResult, trajectory) tuples
        """
        shots = []
        
        for i in range(num_shots):
            shot_result, trajectory = self.generate_single_shot(shot_type, i)
            shots.append((shot_result, trajectory))
            
            if progress_callback:
                progress_callback(i + 1, num_shots, shot_type)
        
        return shots
    
    def save_shots_to_csv(self, shots_data: List[Tuple[str, ShotResult, List[TrajectoryPoint]]], 
                         filename: str) -> str:
        """
        Save shot data to CSV file.
        
        Args:
            shots_data: List of (shot_type, shot_result, trajectory) tuples
            filename: Output filename
        
        Returns:
            Full path to saved file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['shot_type', 'shot_id', 'time', 'x', 'y', 'z', 
                           'velocity', 'force', 'energy', 'parameters'])
            
            shot_id = 0
            for shot_type, shot_result, trajectory in shots_data:
                # Write trajectory points
                for point in trajectory:
                    writer.writerow([
                        shot_type,
                        shot_id,
                        f"{point.time:.3f}",
                        f"{point.x:.3f}",
                        f"{point.y:.3f}",
                        f"{point.z:.3f}",
                        f"{shot_result.velocity:.2f}",
                        f"{shot_result.force:.1f}",
                        f"{shot_result.energy:.2f}",
                        str(shot_result.parameters)
                    ])
                shot_id += 1
        
        return filepath
    
    def generate_batch(self, shot_types: List[str], shots_per_type: int, 
                      output_filename: Optional[str] = None) -> str:
        """
        Generate a batch of shots for multiple types and save to CSV.
        
        Args:
            shot_types: List of shot types to generate
            shots_per_type: Number of shots per type
            output_filename: Custom output filename (auto-generated if None)
        
        Returns:
            Path to saved CSV file
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"hockey_shots_{timestamp}.csv"
        
        all_shots_data = []
        total_shots = len(shot_types) * shots_per_type
        current_shot = 0
        
        print(f"🏑 Generating {total_shots} shots ({shots_per_type} per type)")
        print("=" * 50)
        
        for shot_type in shot_types:
            print(f"Generating {shot_type} shots...")
            
            def progress_callback(completed, total, stype):
                nonlocal current_shot
                current_shot += 1
                percent = (current_shot / total_shots) * 100
                print(f"  {stype}: {completed}/{total} ({percent:.1f}% overall)")
            
            shots = self.generate_multiple_shots(shot_type, shots_per_type, progress_callback)
            
            # Add to combined data
            for shot_result, trajectory in shots:
                all_shots_data.append((shot_type, shot_result, trajectory))
        
        # Save to CSV
        filepath = self.save_shots_to_csv(all_shots_data, output_filename)
        
        print(f"\n✅ Generated {total_shots} shots saved to: {filepath}")
        print(f"📊 File size: {os.path.getsize(filepath) / 1024:.1f} KB")
        
        return filepath


def print_generation_summary(filepath: str):
    """Print summary of generated data"""
    # Count shots by type
    shot_counts = {}
    total_points = 0
    
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            shot_type = row['shot_type']
            shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
            total_points += 1
    
    print(f"\n📈 Generation Summary:")
    print("-" * 30)
    for shot_type, count in shot_counts.items():
        unique_shots = len(set())  # This would need shot_id tracking
        print(f"{shot_type:12}: {count:,} trajectory points")
    
    print(f"{'Total':12}: {total_points:,} trajectory points")


def main():
    """Main execution function with argument parsing"""
    parser = argparse.ArgumentParser(description="Generate procedural hockey shots")
    
    parser.add_argument("--shots", type=int, default=1000,
                       help="Number of shots per type (default: 1000)")
    
    parser.add_argument("--types", type=str, default="all",
                       help="Shot types to generate (comma-separated or 'all')")
    
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename (auto-generated if not specified)")
    
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory (default: data)")
    
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible results")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}")
    
    # Parse shot types
    if args.types.lower() == "all":
        shot_types = get_all_shot_types()
    else:
        shot_types = [shot_type_from_string(t.strip()) for t in args.types.split(",")]
    
    print(f"🎯 Shot types: {', '.join(shot_types)}")
    print(f"📊 Shots per type: {args.shots}")
    
    # Generate shots
    generator = ShotGenerator(args.output_dir)
    filepath = generator.generate_batch(shot_types, args.shots, args.output)
    
    # Print summary
    print_generation_summary(filepath)
    
    print(f"\n🚀 Ready for visualization!")
    print(f"   Use: python3 plot.py --file {os.path.basename(filepath)}")


if __name__ == "__main__":
    main()
