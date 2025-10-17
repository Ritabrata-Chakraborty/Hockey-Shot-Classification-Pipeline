"""
Feature extraction from hockey shot trajectories.
Extracts comprehensive features from 3D trajectory data:
- 600 temporal features: 200 timesteps × 3 coordinates (x,y,z)
- 52 auxiliary features: aggregated statistics
Total: 652 features for ML models
"""

import numpy as np
from typing import List
from ..physics.shot_logical import TrajectoryPoint


def extract_temporal_features(trajectory: List[TrajectoryPoint], max_length: int = 200) -> np.ndarray:
    """
    Extract temporal features: x,y,z coordinates over time.
    
    Args:
        trajectory: List of TrajectoryPoint objects
        max_length: Number of timesteps to resample to
        
    Returns:
        (600,) array: flattened [x1, y1, z1, x2, y2, z2, ..., x200, y200, z200]
    """
    if len(trajectory) < 2:
        return np.zeros(max_length * 3)
    
    # Extract coordinates
    coords = np.array([[p.x, p.y, p.z] for p in trajectory])  # [timesteps, 3]
    
    # Resample to max_length timesteps
    if len(coords) > max_length:
        indices = np.linspace(0, len(coords)-1, max_length, dtype=int)
        coords = coords[indices]
    else:
        # Pad with last value
        padding = np.repeat([coords[-1]], max_length - len(coords), axis=0)
        coords = np.vstack([coords, padding])
    
    return coords.flatten()  # [600,]


def extract_temporal_features_aligned(trajectory: List[TrajectoryPoint], max_length: int = 200) -> np.ndarray:
    """
    Extract temporal features with apex-centered alignment (same as TCN).
    
    Args:
        trajectory: List of TrajectoryPoint objects
        max_length: Number of timesteps to resample to
        
    Returns:
        (600,) array: flattened apex-aligned [x1, y1, z1, ..., x200, y200, z200]
    """
    if len(trajectory) < 2:
        return np.zeros(max_length * 3)
    
    coords = np.array([[p.x, p.y, p.z] for p in trajectory])
    
    # Find apex (max height) for temporal alignment
    apex_idx = np.argmax(coords[:, 2])
    
    # Split into ascent/descent phases
    ascent = coords[:apex_idx+1]
    descent = coords[apex_idx+1:]
    
    # Resample each phase to half length
    half_length = max_length // 2
    
    if len(ascent) > half_length:
        indices_asc = np.linspace(0, len(ascent)-1, half_length, dtype=int)
        ascent = ascent[indices_asc]
    else:
        padding = np.repeat([ascent[0]], half_length - len(ascent), axis=0)
        ascent = np.vstack([padding, ascent])
    
    if len(descent) > half_length:
        indices_desc = np.linspace(0, len(descent)-1, half_length, dtype=int)
        descent = descent[indices_desc]
    else:
        padding = np.repeat([descent[-1] if len(descent) > 0 else ascent[-1]], 
                          half_length - len(descent), axis=0)
        descent = np.vstack([descent, padding])
    
    # Combine
    coords = np.vstack([ascent, descent])
    
    return coords.flatten()


def extract_auxiliary_features(trajectory: List[TrajectoryPoint]) -> np.ndarray:
    """
    Extract comprehensive features from trajectory.
    
    Returns: (52,) feature vector
    
    Features grouped by category:
    - Spatial (10): max_range, max_height, path_length, etc.
    - Temporal (5): flight_time, ascent_time, descent_time, etc.
    - Velocity (8): initial/max/final/avg speed, etc.
    - Acceleration (6): max/avg acceleration, etc.
    - Angular (6): launch/landing angle, curvature, etc.
    - Energy (4): kinetic/potential energy, etc.
    - Shape (8): curvature, symmetry, apex position, etc.
    - Statistical (5): entropy, smoothness, variance, etc.
    """
    
    if len(trajectory) < 2:
        return np.zeros(52)
    
    features = []
    
    # Extract basic arrays
    times = np.array([p.time for p in trajectory])
    x_coords = np.array([p.x for p in trajectory])
    y_coords = np.array([p.y for p in trajectory])
    z_coords = np.array([p.z for p in trajectory])
    
    # === SPATIAL FEATURES (10) ===
    max_range = np.max(x_coords)
    max_height = np.max(z_coords)
    max_lateral = np.max(np.abs(y_coords))
    landing_distance = x_coords[-1]
    landing_height = z_coords[-1]
    
    # Path length
    path_length = 0.0
    for i in range(1, len(trajectory)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        dz = z_coords[i] - z_coords[i-1]
        path_length += np.sqrt(dx**2 + dy**2 + dz**2)
    
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    z_range = np.max(z_coords) - np.min(z_coords)
    volume_enclosed = x_range * y_range * z_range
    
    features.extend([max_range, max_height, max_lateral, landing_distance, 
                    landing_height, path_length, x_range, y_range, z_range, volume_enclosed])
    
    # === TEMPORAL FEATURES (5) ===
    flight_time = times[-1] - times[0]
    apex_idx = np.argmax(z_coords)
    time_to_peak = times[apex_idx] - times[0]
    ascent_time = time_to_peak
    descent_time = flight_time - ascent_time
    time_ratio = ascent_time / (flight_time + 1e-8)
    
    features.extend([flight_time, time_to_peak, ascent_time, descent_time, time_ratio])
    
    # === VELOCITY FEATURES (8) ===
    velocities = []
    for i in range(1, len(trajectory)):
        dt = times[i] - times[i-1]
        if dt > 0:
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            dz = z_coords[i] - z_coords[i-1]
            v = np.sqrt(dx**2 + dy**2 + dz**2) / dt
            velocities.append(v)
    
    velocities = np.array(velocities) if velocities else np.array([0.0])
    initial_speed = velocities[0] if len(velocities) > 0 else 0.0
    max_speed = np.max(velocities)
    final_speed = velocities[-1] if len(velocities) > 0 else 0.0
    avg_speed = np.mean(velocities)
    speed_std = np.std(velocities)
    speed_range = max_speed - np.min(velocities)
    
    # Vertical velocities
    vz_initial = (z_coords[1] - z_coords[0]) / (times[1] - times[0]) if len(trajectory) > 1 else 0.0
    vz_final = (z_coords[-1] - z_coords[-2]) / (times[-1] - times[-2]) if len(trajectory) > 1 else 0.0
    
    features.extend([initial_speed, max_speed, final_speed, avg_speed, 
                    speed_std, speed_range, vz_initial, vz_final])
    
    # === ACCELERATION FEATURES (6) ===
    accelerations = []
    if len(velocities) > 1:
        for i in range(1, len(velocities)):
            da = velocities[i] - velocities[i-1]
            dt = times[i+1] - times[i]
            if dt > 0:
                accelerations.append(da / dt)
    
    accelerations = np.array(accelerations) if accelerations else np.array([0.0])
    max_acceleration = np.max(np.abs(accelerations))
    avg_acceleration = np.mean(accelerations)
    max_deceleration = np.min(accelerations) if len(accelerations) > 0 else 0.0
    acceleration_changes = np.sum(np.abs(np.diff(accelerations))) if len(accelerations) > 1 else 0.0
    
    # Vertical and lateral acceleration
    vertical_accel_avg = -9.81  # Approximate gravity
    lateral_accel_avg = np.mean(np.abs(np.diff(y_coords))) / (flight_time + 1e-8)
    
    features.extend([max_acceleration, avg_acceleration, max_deceleration, 
                    acceleration_changes, vertical_accel_avg, lateral_accel_avg])
    
    # === ANGULAR FEATURES (6) ===
    # Launch angle
    if len(trajectory) > 1:
        dx0 = x_coords[1] - x_coords[0]
        dz0 = z_coords[1] - z_coords[0]
        launch_angle = np.arctan2(dz0, dx0 + 1e-8)
    else:
        launch_angle = 0.0
    
    # Landing angle
    if len(trajectory) > 1:
        dx_end = x_coords[-1] - x_coords[-2]
        dz_end = z_coords[-1] - z_coords[-2]
        landing_angle = np.arctan2(dz_end, dx_end + 1e-8)
    else:
        landing_angle = 0.0
    
    max_elevation_angle = np.max([np.arctan2(z_coords[i] - z_coords[i-1], 
                                              x_coords[i] - x_coords[i-1] + 1e-8) 
                                  for i in range(1, len(trajectory))])
    avg_elevation_angle = np.mean([np.arctan2(z_coords[i] - z_coords[i-1], 
                                               x_coords[i] - x_coords[i-1] + 1e-8) 
                                   for i in range(1, len(trajectory))])
    azimuth_variance = np.var(y_coords)
    
    # Trajectory curvature
    curvatures = []
    for i in range(2, len(trajectory)):
        v1 = np.array([x_coords[i-1] - x_coords[i-2], 
                      y_coords[i-1] - y_coords[i-2], 
                      z_coords[i-1] - z_coords[i-2]])
        v2 = np.array([x_coords[i] - x_coords[i-1], 
                      y_coords[i] - y_coords[i-1], 
                      z_coords[i] - z_coords[i-1]])
        cross = np.cross(v1, v2)
        curvature = np.linalg.norm(cross) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        curvatures.append(curvature)
    
    avg_curvature = np.mean(curvatures) if curvatures else 0.0
    
    features.extend([launch_angle, landing_angle, max_elevation_angle, 
                    avg_elevation_angle, azimuth_variance, avg_curvature])
    
    # === ENERGY FEATURES (4) ===
    M_BALL = 0.156  # kg
    initial_ke = 0.5 * M_BALL * initial_speed**2
    max_pe = M_BALL * 9.81 * max_height
    total_energy = initial_ke + max_pe
    energy_loss_rate = (initial_ke - 0.5 * M_BALL * final_speed**2) / (flight_time + 1e-8)
    
    features.extend([initial_ke, max_pe, total_energy, energy_loss_rate])
    
    # === SHAPE FEATURES (8) ===
    max_curvature = np.max(curvatures) if curvatures else 0.0
    
    # Straightness (ratio of straight line to path length)
    straight_line_dist = np.sqrt((x_coords[-1] - x_coords[0])**2 + 
                                 (y_coords[-1] - y_coords[0])**2 + 
                                 (z_coords[-1] - z_coords[0])**2)
    straightness = straight_line_dist / (path_length + 1e-8)
    
    # Spin rate estimate (change in curvature)
    spin_rate = np.std(curvatures) if len(curvatures) > 1 else 0.0
    
    # Wobble factor
    wobble = np.std(y_coords) / (max_range + 1e-8)
    
    # Apex position ratio
    apex_position_ratio = times[apex_idx] / (flight_time + 1e-8)
    
    # Symmetry score (compare ascent and descent)
    symmetry_score = 1.0 - abs(ascent_time - descent_time) / (flight_time + 1e-8)
    
    # Arc completeness (how parabolic)
    expected_apex_time = flight_time / 2.0
    arc_completeness = 1.0 - abs(time_to_peak - expected_apex_time) / (flight_time + 1e-8)
    
    # Vertical displacement efficiency
    vertical_efficiency = max_height / (path_length + 1e-8)
    
    features.extend([max_curvature, straightness, spin_rate, wobble, 
                    apex_position_ratio, symmetry_score, arc_completeness, vertical_efficiency])
    
    # === STATISTICAL FEATURES (5) ===
    # Position entropy (spread of positions)
    x_entropy = -np.sum((x_coords / (np.sum(x_coords) + 1e-8)) * 
                       np.log((x_coords / (np.sum(x_coords) + 1e-8)) + 1e-8))
    
    # Velocity entropy
    vel_probs = velocities / (np.sum(velocities) + 1e-8)
    velocity_entropy = -np.sum(vel_probs * np.log(vel_probs + 1e-8))
    
    # Smoothness (jerk - derivative of acceleration)
    if len(accelerations) > 1:
        jerk = np.diff(accelerations)
        smoothness = 1.0 / (np.mean(np.abs(jerk)) + 1e-8)
    else:
        smoothness = 1.0
    
    variance_x = np.var(x_coords)
    variance_z = np.var(z_coords)
    
    features.extend([x_entropy, velocity_entropy, smoothness, variance_x, variance_z])
    
    return np.array(features, dtype=np.float32)


def extract_all_features(trajectory: List[TrajectoryPoint], max_length: int = 200, 
                        aligned: bool = True) -> np.ndarray:
    """
    Extract all 652 features: 600 temporal + 52 auxiliary.
    
    Args:
        trajectory: List of TrajectoryPoint objects
        max_length: Number of timesteps for temporal features
        aligned: If True, use apex-centered alignment (recommended for consistency with TCN)
        
    Returns:
        (652,) array: [temporal_600, auxiliary_52]
    """
    if aligned:
        temporal = extract_temporal_features_aligned(trajectory, max_length)  # 600
    else:
        temporal = extract_temporal_features(trajectory, max_length)  # 600
    auxiliary = extract_auxiliary_features(trajectory)  # 52
    return np.concatenate([temporal, auxiliary])  # 652


def extract_trajectory_features(trajectory: List[TrajectoryPoint]) -> np.ndarray:
    """
    Unified feature extraction: returns all 652 features for ML models.
    For backwards compatibility, this now returns 652 features.
    """
    return extract_all_features(trajectory)

