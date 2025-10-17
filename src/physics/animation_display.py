import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .shot_logical import TrajectoryPoint

@dataclass
class AnimationConfig:
    figure_size: Tuple[int, int] = (12, 8)
    playback_speed: float = 1.0
    trail_length: int = 50
    show_field: bool = True

class HockeyField3D:
    def __init__(self):
        self.field_length = 91.4
        self.field_width = 55.0
        self.goal_width = 3.66
        self.goal_height = 2.14
        
    def draw_field(self, ax):
        field_x = [0, self.field_length, self.field_length, 0, 0]
        field_y = [-self.field_width/2, -self.field_width/2, self.field_width/2, self.field_width/2, -self.field_width/2]
        ax.plot(field_x, field_y, [0]*5, 'k-', linewidth=2, alpha=0.8)
        
        goal_x = self.field_length
        goal_y1, goal_y2 = -self.goal_width/2, self.goal_width/2
        ax.plot([goal_x, goal_x], [goal_y1, goal_y1], [0, self.goal_height], 'r-', linewidth=3)
        ax.plot([goal_x, goal_x], [goal_y2, goal_y2], [0, self.goal_height], 'r-', linewidth=3)
        ax.plot([goal_x, goal_x], [goal_y1, goal_y2], [self.goal_height, self.goal_height], 'r-', linewidth=3)

class TrajectoryAnimator:
    def __init__(self, config: AnimationConfig = None):
        self.config = config or AnimationConfig()
        self.trajectories = []
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
        
    def add_trajectory(self, trajectory: List[TrajectoryPoint], label: str = ""):
        color = self.colors[len(self.trajectories) % len(self.colors)]
        self.trajectories.append({'points': trajectory, 'label': label, 'color': color})
        
    def animate(self, save_gif: str = None):
        if not self.trajectories:
            return
            
        fig = plt.figure(figsize=self.config.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        if self.config.show_field:
            HockeyField3D().draw_field(ax)
        
        all_x = [p.x for traj in self.trajectories for p in traj['points']]
        all_y = [p.y for traj in self.trajectories for p in traj['points']]
        all_z = [p.z for traj in self.trajectories for p in traj['points']]
        
        ax.set_xlim(0, max(all_x) * 1.1 if all_x else 100)
        ax.set_ylim(-max(abs(min(all_y)), max(all_y)) * 1.1 if all_y else 30, 
                   max(abs(min(all_y)), max(all_y)) * 1.1 if all_y else 30)
        ax.set_zlim(0, max(all_z) * 1.1 if all_z else 20)
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Width (m)')
        ax.set_zlabel('Height (m)')
        ax.set_title('Hockey Shot Trajectory Animation')
        
        lines = []
        balls = []
        for traj in self.trajectories:
            line, = ax.plot([], [], [], color=traj['color'], linewidth=2, label=traj['label'])
            ball, = ax.plot([], [], [], 'o', color=traj['color'], markersize=8)
            lines.append(line)
            balls.append(ball)
        
        if any(traj['label'] for traj in self.trajectories):
            ax.legend()
        
        max_frames = max(len(traj['points']) for traj in self.trajectories)
        
        def animate_frame(frame):
            for i, (traj, line, ball) in enumerate(zip(self.trajectories, lines, balls)):
                if frame < len(traj['points']):
                    points = traj['points'][:frame+1]
                    x_data = [p.x for p in points]
                    y_data = [p.y for p in points]
                    z_data = [p.z for p in points]
                    
                    line.set_data(x_data, y_data)
                    line.set_3d_properties(z_data)
                    
                    if points:
                        ball.set_data([points[-1].x], [points[-1].y])
                        ball.set_3d_properties([points[-1].z])
            return lines + balls
        
        anim = animation.FuncAnimation(fig, animate_frame, frames=max_frames, 
                                     interval=int(50/self.config.playback_speed), 
                                     blit=False, repeat=True)
        
        if save_gif:
            anim.save(save_gif, writer='pillow', fps=20)
        
        plt.show()

def animate_single_trajectory(trajectory: List[TrajectoryPoint], title: str = "Hockey Shot"):
    animator = TrajectoryAnimator()
    animator.add_trajectory(trajectory, title)
    animator.animate()

def animate_trajectory_comparison(trajectories: Dict[str, List[TrajectoryPoint]], 
                                title: str = "Shot Comparison"):
    animator = TrajectoryAnimator()
    for label, trajectory in trajectories.items():
        animator.add_trajectory(trajectory, label)
    animator.animate()

def create_trajectory_gif(trajectories: Dict[str, List[TrajectoryPoint]], 
                         output_path: str, 
                         title: str = "Hockey Shot Types Comparison",
                         fps: int = 10,
                         dpi: int = 100) -> bool:
    """
    Create an animated GIF showing trajectory progression without field lines.
    
    Args:
        trajectories: Dictionary mapping shot type names to trajectory point lists
        output_path: Path where to save the GIF file
        title: Title for the plot
        fps: Frames per second for the animation
        dpi: Resolution of the output GIF
        
    Returns:
        True if GIF was created successfully, False otherwise
    """
    try:
        if not trajectories:
            return False
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Prepare trajectory data
        trajectories_data = []
        max_points = 0
        
        for i, (label, trajectory) in enumerate(trajectories.items()):
            x_coords = [p.x for p in trajectory]
            y_coords = [p.y for p in trajectory]
            z_coords = [p.z for p in trajectory]
            
            trajectories_data.append({
                'label': label,
                'x': x_coords,
                'y': y_coords,
                'z': z_coords,
                'color': colors[i % len(colors)]
            })
            max_points = max(max_points, len(trajectory))
        
        # Set up plot limits
        all_x = [p.x for traj in trajectories.values() for p in traj]
        all_y = [p.y for traj in trajectories.values() for p in traj]
        all_z = [p.z for traj in trajectories.values() for p in traj]
        
        if all_x and all_y and all_z:
            ax.set_xlim(0, max(all_x) * 1.1)
            ax.set_ylim(min(all_y) * 1.2, max(all_y) * 1.2)
            ax.set_zlim(0, max(all_z) * 1.2)
        
        # Set labels and title
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Lateral (m)')
        ax.set_zlabel('Height (m)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set clean background
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        
        # Initialize line objects for animation
        lines = []
        points = []
        
        for traj_data in trajectories_data:
            line, = ax.plot([], [], [], color=traj_data['color'], 
                           linewidth=2, alpha=0.8, label=traj_data['label'])
            point = ax.scatter([], [], [], color=traj_data['color'], 
                             s=120, alpha=1.0, marker='o', edgecolors='black', linewidth=1)
            lines.append(line)
            points.append(point)
        
        ax.legend(loc='upper right')
        
        def animate_frame(frame):
            for i, (line, point, traj_data) in enumerate(zip(lines, points, trajectories_data)):
                if frame < len(traj_data['x']):
                    # Show full trajectory from start to current frame
                    x_full = traj_data['x'][:frame + 1]
                    y_full = traj_data['y'][:frame + 1]
                    z_full = traj_data['z'][:frame + 1]
                    
                    line.set_data(x_full, y_full)
                    line.set_3d_properties(z_full)
                    
                    # Current point (moving ball)
                    point._offsets3d = ([traj_data['x'][frame]], 
                                      [traj_data['y'][frame]], 
                                      [traj_data['z'][frame]])
                else:
                    # Show complete trajectory when animation is done
                    line.set_data(traj_data['x'], traj_data['y'])
                    line.set_3d_properties(traj_data['z'])
                    
                    # Show end point
                    if traj_data['x']:
                        point._offsets3d = ([traj_data['x'][-1]], 
                                          [traj_data['y'][-1]], 
                                          [traj_data['z'][-1]])
            
            return lines + points
        
        # Create animation
        frames = max_points + 30  # Extra frames to show complete trajectories
        anim = animation.FuncAnimation(fig, animate_frame, frames=frames, 
                                     interval=int(1000/fps), blit=False, repeat=True)
        
        # Save as GIF with fallback options
        try:
            anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
            plt.close()
            return True
        except Exception as gif_error:
            try:
                anim.save(output_path, writer='imagemagick', fps=fps, dpi=dpi)
                plt.close()
                return True
            except Exception as img_error:
                # Fallback to PNG
                png_path = output_path.replace('.gif', '.png')
                plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
                plt.close()
                return False
                
    except Exception as e:
        plt.close()
        return False

# CLI support
if __name__ == '__main__':
    import sys, os, csv
    from collections import defaultdict
    from .shot_logical import TrajectoryPoint
    from .statistical_plots import create_comprehensive_analysis
    
    if len(sys.argv) != 3:
        print("Usage: python -m src.physics.animation_display <csv_file> <output_dir>")
        sys.exit(1)
    
    csv_file, output_dir = sys.argv[1], sys.argv[2]
    
    # Load CSV
    trajectories, current_shot = defaultdict(list), {}
    with open(csv_file, 'r') as f:
        for row in csv.DictReader(f):
            key = (row['shot_type'], int(row['shot_id']))
            if key not in current_shot:
                current_shot[key] = []
            current_shot[key].append(TrajectoryPoint(
                float(row['time']), float(row['x']), float(row['y']), float(row['z'])))
    
    for (shot_type, _), points in current_shot.items():
        trajectories[shot_type].append(sorted(points, key=lambda p: p.time))
    
    # Create animation
    os.makedirs(f'{output_dir}/animations', exist_ok=True)
    comparison_data = {f'{k.title()} Shot': v[0] for k, v in trajectories.items() if v}
    if comparison_data:
        success = create_trajectory_gif(comparison_data, 
                                       f'{output_dir}/animations/shot_types_comparison.gif')
        print('✅ GIF saved' if success else '✅ PNG saved')
    
    # Create analysis
    try:
        os.makedirs(f'{output_dir}/analysis_results', exist_ok=True)
        report = create_comprehensive_analysis(dict(trajectories), 
                                               f'{output_dir}/analysis_results', 
                                               show_plots=False)
        with open(f'{output_dir}/analysis_results/analysis_report.txt', 'w') as f:
            f.write(report)
        print('✅ Analysis complete')
    except Exception as e:
        print(f'⚠️ Analysis failed: {e}')