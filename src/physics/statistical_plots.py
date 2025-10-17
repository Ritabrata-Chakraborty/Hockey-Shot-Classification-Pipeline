import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import scipy.stats as stats
from .shot_logical import TrajectoryPoint

@dataclass
class ShotStatistics:
    shot_type: str
    shot_id: int
    max_range: float
    max_height: float
    flight_time: float
    average_velocity: float
    peak_velocity: float
    energy: float

class ShotAnalyzer:
    def __init__(self):
        self.shot_data = []
        
    def add_shot_data(self, shot_type: str, trajectory: List[TrajectoryPoint], shot_id: int = 0):
        stats = self._calculate_shot_statistics(shot_type, trajectory, shot_id)
        self.shot_data.append(stats)
    
    def _calculate_shot_statistics(self, shot_type: str, trajectory: List[TrajectoryPoint], shot_id: int) -> ShotStatistics:
        if not trajectory:
            return ShotStatistics(shot_type, shot_id, 0, 0, 0, 0, 0, 0)
        
        max_range = max(point.x for point in trajectory)
        max_height = max(point.z for point in trajectory)
        flight_time = trajectory[-1].time
        
        velocities = []
        for i in range(1, len(trajectory)):
            dt = trajectory[i].time - trajectory[i-1].time
            if dt > 0:
                dx = trajectory[i].x - trajectory[i-1].x
                dy = trajectory[i].y - trajectory[i-1].y
                dz = trajectory[i].z - trajectory[i-1].z
                velocity = np.sqrt(dx**2 + dy**2 + dz**2) / dt
                velocities.append(velocity)
        
        average_velocity = np.mean(velocities) if velocities else 0
        peak_velocity = max(velocities) if velocities else 0
        energy = 0.5 * 0.156 * peak_velocity**2
        
        return ShotStatistics(shot_type, shot_id, max_range, max_height, 
                            flight_time, average_velocity, peak_velocity, energy)
    
    def get_dataframe(self) -> pd.DataFrame:
        data = []
        for stat in self.shot_data:
            data.append({
                'shot_type': stat.shot_type,
                'shot_id': stat.shot_id,
                'max_range': stat.max_range,
                'max_height': stat.max_height,
                'flight_time': stat.flight_time,
                'average_velocity': stat.average_velocity,
                'peak_velocity': stat.peak_velocity,
                'energy': stat.energy
            })
        return pd.DataFrame(data)

class StatisticalPlotter:
    def __init__(self, analyzer: ShotAnalyzer):
        self.analyzer = analyzer
        self.df = analyzer.get_dataframe()
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_performance_overview(self, figsize: Tuple[int, int] = (15, 10)):
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Hockey Shot Performance Overview', fontsize=16, fontweight='bold')
        
        sns.boxplot(data=self.df, x='shot_type', y='max_range', ax=axes[0, 0])
        axes[0, 0].set_title('Range Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=self.df, x='shot_type', y='max_height', ax=axes[0, 1])
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=self.df, x='shot_type', y='peak_velocity', ax=axes[0, 2])
        axes[0, 2].set_title('Peak Velocity Distribution')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=self.df, x='shot_type', y='flight_time', ax=axes[1, 0])
        axes[1, 0].set_title('Flight Time Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=self.df, x='shot_type', y='energy', ax=axes[1, 1])
        axes[1, 1].set_title('Energy Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for shot_type in self.df['shot_type'].unique():
            data = self.df[self.df['shot_type'] == shot_type]
            axes[1, 2].scatter(data['max_range'], data['max_height'], 
                             label=shot_type, alpha=0.7, s=50)
        axes[1, 2].set_title('Range vs Height')
        axes[1, 2].set_xlabel('Max Range (m)')
        axes[1, 2].set_ylabel('Max Height (m)')
        axes[1, 2].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_statistical_comparison(self, figsize: Tuple[int, int] = (12, 8)):
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Statistical Analysis', fontsize=16, fontweight='bold')
        
        numeric_cols = ['max_range', 'max_height', 'flight_time', 'peak_velocity', 'energy']
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
        axes[0, 0].set_title('Correlation Matrix')
        
        for shot_type in self.df['shot_type'].unique():
            data = self.df[self.df['shot_type'] == shot_type]['max_range']
            sns.histplot(data, alpha=0.6, label=shot_type, ax=axes[0, 1], kde=True)
        axes[0, 1].set_title('Range Distribution')
        axes[0, 1].legend()
        
        sns.violinplot(data=self.df, x='shot_type', y='peak_velocity', ax=axes[1, 0])
        axes[1, 0].set_title('Velocity Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        shot_types = self.df['shot_type'].unique()
        metrics = ['max_range', 'max_height', 'peak_velocity', 'energy']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 4, projection='polar')
        for shot_type in shot_types:
            data = self.df[self.df['shot_type'] == shot_type]
            values = []
            for metric in metrics:
                max_val = self.df[metric].max()
                min_val = self.df[metric].min()
                normalized = (data[metric].mean() - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                values.append(normalized)
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=shot_type)
            ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_title('Performance Profile')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> str:
        report = ["HOCKEY SHOT STATISTICAL ANALYSIS", "="*50, ""]
        report.append(f"Total shots: {len(self.df)}")
        report.append(f"Shot types: {', '.join(self.df['shot_type'].unique())}")
        report.append("")
        
        for shot_type in self.df['shot_type'].unique():
            data = self.df[self.df['shot_type'] == shot_type]
            report.append(f"{shot_type.upper()}:")
            report.append(f"  Count: {len(data)}")
            report.append(f"  Avg Range: {data['max_range'].mean():.1f}±{data['max_range'].std():.1f}m")
            report.append(f"  Avg Height: {data['max_height'].mean():.1f}±{data['max_height'].std():.1f}m")
            report.append(f"  Avg Velocity: {data['peak_velocity'].mean():.1f}±{data['peak_velocity'].std():.1f}m/s")
            report.append("")
        
        return "\n".join(report)
    
    def save_all_plots(self, output_dir: str = "plots"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig1 = self.plot_performance_overview()
        fig1.savefig(f"{output_dir}/performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        fig2 = self.plot_statistical_comparison()
        fig2.savefig(f"{output_dir}/statistical_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        with open(f"{output_dir}/statistical_report.txt", 'w') as f:
            f.write(self.generate_report())

def analyze_shot_data(trajectories: Dict[str, List[List[TrajectoryPoint]]]) -> ShotAnalyzer:
    analyzer = ShotAnalyzer()
    for shot_type, trajectory_list in trajectories.items():
        for i, trajectory in enumerate(trajectory_list):
            analyzer.add_shot_data(shot_type, trajectory, i)
    return analyzer

def create_comprehensive_analysis(trajectories: Dict[str, List[List[TrajectoryPoint]]], 
                                output_dir: str = "analysis_output", show_plots: bool = True) -> str:
    analyzer = analyze_shot_data(trajectories)
    plotter = StatisticalPlotter(analyzer)
    
    if show_plots:
        plotter.plot_performance_overview()
        plt.show()
        plotter.plot_statistical_comparison()
        plt.show()
    
    plotter.save_all_plots(output_dir)
    return plotter.generate_report()


if __name__ == '__main__':
    import sys
    import csv
    from collections import defaultdict
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.physics.statistical_plots <csv_file> [output_dir]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'analysis/statistics'
    
    print(f"Loading data from: {csv_file}")
    
    # Load CSV data
    trajectories = defaultdict(list)
    current_shot = {}
    
    with open(csv_file, 'r') as f:
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
    
    print(f"Loaded {sum(len(v) for v in trajectories.values())} trajectories")
    print(f"Shot types: {list(trajectories.keys())}")
    
    # Create analysis
    print(f"\nGenerating statistical analysis...")
    report = create_comprehensive_analysis(trajectories, output_dir, show_plots=False)
    
    # Save report
    report_file = f"{output_dir}/statistical_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Analysis complete!")
    print(f"   Plots saved to: {output_dir}/")
    print(f"   Report saved to: {report_file}")