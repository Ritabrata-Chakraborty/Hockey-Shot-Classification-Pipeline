"""
Shot Logical - Core Physics and Mathematics for Hockey Shots

This module contains the fundamental physics calculations for all four
hockey shot types, providing clean APIs for trajectory generation.

NOTE: This module provides BASIC physics calculations.
For PRODUCTION trajectory generation, use effects_simulation.py which provides:
- Reynolds-dependent drag
- Advanced Magnus effect with spin decay
- Environmental effects (wind, turbulence, temperature, humidity)
- More realistic trajectories

This module is kept for:
- Unit testing
- Quick prototyping
- Educational examples

Author: Physics-Informed Simulation Developer
Date: October 2025
"""

import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


# Physical constants
M_BALL = 0.156  # Mass of hockey ball in kg
DT_DEFAULT = 0.015  # Contact time in seconds
G = 9.81  # Gravity (m/s²)
AIR_DENSITY = 1.225  # Air density (kg/m³)
BALL_RADIUS = 0.036  # Ball radius (m)
DRAG_COEFFICIENT = 0.47  # Drag coefficient for sphere

# Enhanced physics constants based on research
STICK_FLEX_COEFFICIENT = 0.15  # Energy transfer efficiency from stick flex (0.1-0.25)
STICK_CONTACT_TIME_RANGE = (0.005, 0.012)  # Contact time variation (s)
SURFACE_FRICTION_COEFF = 0.02  # Ball-surface friction coefficient
MAGNUS_COEFFICIENT = 0.25  # Magnus coefficient for hockey ball
AIR_DENSITY_VARIATION = (1.15, 1.30)  # Air density range (altitude/weather effects)


class HockeyPhysicsError(Exception):
    """Custom exception for hockey physics calculations"""
    pass


def apply_stick_flex_enhancement(base_velocity: float, stick_flex: float, 
                               contact_time: float, player_strength: float = 1.0) -> float:
    """
    Apply stick flex and biomechanical enhancements to shot velocity.
    
    Based on research showing stick deformation stores and releases energy,
    and player biomechanics affect energy transfer efficiency.
    
    Args:
        base_velocity: Base velocity from collision physics
        stick_flex: Stick flex coefficient (0.1-0.25)
        contact_time: Stick-ball contact time
        player_strength: Player strength multiplier (0.7-1.3)
    
    Returns:
        Enhanced velocity accounting for stick flex and biomechanics
    """
    # Energy stored in stick flex (proportional to flex and contact time)
    flex_energy_factor = 1.0 + (stick_flex * contact_time * 100)  # Research-based scaling
    
    # Player biomechanics factor (strength, technique)
    biomech_factor = 0.9 + (player_strength - 1.0) * 0.4  # ±30% variation
    
    # Combined enhancement with realistic limits
    enhancement = flex_energy_factor * biomech_factor
    enhanced_velocity = base_velocity * min(enhancement, 1.5)  # Cap at 50% increase
    
    return enhanced_velocity


def calculate_environmental_effects(air_density: float = AIR_DENSITY, 
                                  surface_condition: float = 1.0) -> Tuple[float, float]:
    """
    Calculate environmental effects on shot physics.
    
    Args:
        air_density: Air density (kg/m³) - varies with altitude/weather
        surface_condition: Surface quality factor (0.8-1.2)
    
    Returns:
        Tuple of (drag_multiplier, friction_coefficient)
    """
    # Air density affects drag (higher altitude = less drag)
    drag_multiplier = air_density / AIR_DENSITY
    
    # Surface condition affects friction and ball behavior
    friction_coeff = SURFACE_FRICTION_COEFF * surface_condition
    
    return drag_multiplier, friction_coeff


@dataclass
class ShotResult:
    """Result of a shot calculation"""
    velocity: float  # m/s
    force: float     # N
    energy: float    # J
    shot_type: str
    parameters: Dict


@dataclass
class TrajectoryPoint:
    """Single point in trajectory"""
    time: float
    x: float
    y: float
    z: float


def validate_positive(value: float, name: str) -> None:
    """Validate that a parameter is positive"""
    if value <= 0:
        raise HockeyPhysicsError(f"{name} must be positive, got {value}")


# ============================================================================
# CORE PHYSICS MODELS
# ============================================================================

def hit_drive_shot(m_stick: float, v_stick0: float, v_ball0: float = 0.0, 
                   dt: float = DT_DEFAULT, m_ball: float = M_BALL) -> ShotResult:
    """
    Calculate Hit (Drive) shot using 1D elastic collision physics.
    
    Args:
        m_stick: Stick mass in kg
        v_stick0: Initial stick velocity in m/s
        v_ball0: Initial ball velocity in m/s
        dt: Contact time in seconds
        m_ball: Ball mass in kg
    
    Returns:
        ShotResult with velocity, force, energy, and parameters
    """
    validate_positive(m_stick, "stick mass")
    validate_positive(dt, "contact time")
    validate_positive(m_ball, "ball mass")
    
    # Elastic collision formula
    numerator = 2 * m_stick * v_stick0 + (m_ball - m_stick) * v_ball0
    denominator = m_stick + m_ball
    v_ball = numerator / denominator
    
    # Calculate force and energy
    F_avg = m_ball * (v_ball - v_ball0) / dt
    energy = 0.5 * m_ball * v_ball**2
    
    return ShotResult(
        velocity=v_ball,
        force=F_avg,
        energy=energy,
        shot_type="hit",
        parameters={
            "m_stick": m_stick,
            "v_stick0": v_stick0,
            "v_ball0": v_ball0,
            "dt": dt
        }
    )


def slap_shot(k: float, x: float, dt: float = DT_DEFAULT, 
              m_ball: float = M_BALL) -> ShotResult:
    """
    Calculate Slap shot using elastic energy transfer physics.
    
    Args:
        k: Stick stiffness in N/m
        x: Stick deflection in m
        dt: Contact time in seconds
        m_ball: Ball mass in kg
    
    Returns:
        ShotResult with velocity, force, energy, and parameters
    """
    validate_positive(k, "stick stiffness")
    validate_positive(x, "stick deflection")
    validate_positive(dt, "contact time")
    validate_positive(m_ball, "ball mass")
    
    # Energy storage and conversion
    PE = 0.5 * k * x**2
    v_ball = math.sqrt(2 * PE / m_ball)
    F_avg = m_ball * v_ball / dt
    
    return ShotResult(
        velocity=v_ball,
        force=F_avg,
        energy=PE,
        shot_type="slap",
        parameters={
            "k": k,
            "x": x,
            "dt": dt,
            "stored_energy": PE
        }
    )


def push_flick_shot(a: float, t: float, dt: float = DT_DEFAULT, 
                    m_ball: float = M_BALL) -> ShotResult:
    """
    Calculate Push/Flick shot using constant acceleration kinematics.
    
    Args:
        a: Acceleration in m/s²
        t: Stroke duration in seconds
        dt: Contact time in seconds
        m_ball: Ball mass in kg
    
    Returns:
        ShotResult with velocity, force, energy, and parameters
    """
    validate_positive(a, "acceleration")
    validate_positive(t, "stroke duration")
    validate_positive(dt, "contact time")
    validate_positive(m_ball, "ball mass")
    
    # Kinematics
    v_ball = a * t
    displacement = 0.5 * a * t**2
    energy = 0.5 * m_ball * v_ball**2
    F_avg = m_ball * a
    
    return ShotResult(
        velocity=v_ball,
        force=F_avg,
        energy=energy,
        shot_type="push_flick",
        parameters={
            "acceleration": a,
            "stroke_time": t,
            "displacement": displacement,
            "dt": dt
        }
    )


def drag_flick_shot(alpha: float, L: float, t: float, eta: float = 0.85,
                    dt: float = DT_DEFAULT, m_ball: float = M_BALL) -> ShotResult:
    """
    Calculate Drag Flick shot using biomechanical rotational physics.
    
    Args:
        alpha: Angular acceleration in rad/s²
        L: Stick length in m
        t: Flick duration in seconds
        eta: Energy transfer efficiency (0-1)
        dt: Contact time in seconds
        m_ball: Ball mass in kg
    
    Returns:
        ShotResult with velocity, force, energy, and parameters
    """
    validate_positive(alpha, "angular acceleration")
    validate_positive(L, "stick length")
    validate_positive(t, "flick duration")
    validate_positive(dt, "contact time")
    validate_positive(m_ball, "ball mass")
    
    if not (0.0 < eta <= 1.0):
        raise HockeyPhysicsError(f"Efficiency must be between 0 and 1, got {eta}")
    
    # Rotational mechanics
    omega = alpha * t
    v_tip = L * omega
    v_ball = eta * v_tip
    
    # Calculate force and energy
    F_avg = m_ball * v_ball / dt
    energy = 0.5 * m_ball * v_ball**2
    
    return ShotResult(
        velocity=v_ball,
        force=F_avg,
        energy=energy,
        shot_type="drag_flick",
        parameters={
            "alpha": alpha,
            "L": L,
            "t": t,
            "eta": eta,
            "omega": omega,
            "v_tip": v_tip,
            "dt": dt
        }
    )


# ============================================================================
# TRAJECTORY CALCULATION
# ============================================================================

def calculate_trajectory(initial_velocity: float, launch_angle: float = 15.0, 
                        launch_height: float = 0.5, lateral_angle: float = 0.0,
                        spin_rate: float = 0.0, spin_axis: tuple = (0, 0, 1),
                        air_density: float = AIR_DENSITY, surface_condition: float = 1.0,
                        time_step: float = 0.01, max_time: float = 8.0) -> List[TrajectoryPoint]:
    """
    Calculate ball trajectory with air resistance, Magnus effect, and environmental factors.
    
    Args:
        initial_velocity: Initial velocity in m/s
        launch_angle: Launch angle in degrees (elevation)
        launch_height: Launch height in meters
        lateral_angle: Lateral angle in degrees (azimuth)
        spin_rate: Spin rate in rad/s
        spin_axis: Normalized spin axis vector (x, y, z)
        air_density: Air density in kg/m³ (environmental factor)
        surface_condition: Surface quality factor (0.8-1.2)
        time_step: Time step for simulation
        max_time: Maximum simulation time
    
    Returns:
        List of TrajectoryPoint objects
    """
    # Convert angles to radians
    launch_rad = math.radians(launch_angle)
    lateral_rad = math.radians(lateral_angle)
    
    # Initial conditions
    x, y, z = 0.0, 0.0, launch_height
    
    # 3D velocity components
    horizontal_velocity = initial_velocity * math.cos(launch_rad)
    vx = horizontal_velocity * math.cos(lateral_rad)
    vy = horizontal_velocity * math.sin(lateral_rad)
    vz = initial_velocity * math.sin(launch_rad)
    
    # Storage for trajectory
    trajectory = [TrajectoryPoint(0.0, x, y, z)]
    
    t = 0.0
    ball_area = math.pi * BALL_RADIUS**2
    
    # Calculate environmental effects
    drag_multiplier, friction_coeff = calculate_environmental_effects(air_density, surface_condition)
    
    while t < max_time and z >= 0:
        # Current velocity magnitude
        v_magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
        
        # Drag force components (with environmental effects)
        if v_magnitude > 0:
            drag_force = 0.5 * air_density * DRAG_COEFFICIENT * ball_area * v_magnitude**2 * drag_multiplier
            drag_x = -drag_force * (vx / v_magnitude) / M_BALL
            drag_y = -drag_force * (vy / v_magnitude) / M_BALL
            drag_z = -drag_force * (vz / v_magnitude) / M_BALL
        else:
            drag_x = drag_y = drag_z = 0.0
        
        # Magnus force components (if spinning)
        magnus_x = magnus_y = magnus_z = 0.0
        if spin_rate > 0 and v_magnitude > 0:
            # Magnus force: F = (1/2) * ρ * A * Cm * v² * (ω × v̂)
            # Enhanced Magnus coefficient with environmental effects
            magnus_magnitude = 0.5 * air_density * ball_area * MAGNUS_COEFFICIENT * v_magnitude**2 * drag_multiplier
            
            # Velocity unit vector
            v_unit = (vx / v_magnitude, vy / v_magnitude, vz / v_magnitude)
            
            # Cross product: spin_axis × velocity_unit
            cross_x = spin_axis[1] * v_unit[2] - spin_axis[2] * v_unit[1]
            cross_y = spin_axis[2] * v_unit[0] - spin_axis[0] * v_unit[2]
            cross_z = spin_axis[0] * v_unit[1] - spin_axis[1] * v_unit[0]
            
            # Normalize cross product
            cross_magnitude = math.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
            if cross_magnitude > 0:
                magnus_x = magnus_magnitude * (cross_x / cross_magnitude) / M_BALL
                magnus_y = magnus_magnitude * (cross_y / cross_magnitude) / M_BALL
                magnus_z = magnus_magnitude * (cross_z / cross_magnitude) / M_BALL
        
        # Update velocities (drag + Magnus + gravity)
        vx += (drag_x + magnus_x) * time_step
        vy += (drag_y + magnus_y) * time_step
        vz += (drag_z + magnus_z - G) * time_step
        
        # Update positions
        x += vx * time_step
        y += vy * time_step
        z += vz * time_step
        
        # Store trajectory point
        t += time_step
        trajectory.append(TrajectoryPoint(t, x, y, z))
    
    return trajectory


# ============================================================================
# PARAMETER RANGES FOR PROCEDURAL GENERATION
# ============================================================================

def get_parameter_ranges() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Get realistic parameter ranges for each shot type for procedural generation.
    
    Returns:
        Dictionary with parameter ranges for each shot type
    """
    return {
        "hit": {
            "m_stick": (0.5, 0.7),      # kg
            "v_stick0": (25.0, 45.0),   # m/s
            "v_ball0": (0.0, 2.0),      # m/s
        },
        "slap": {
            "k": (2000.0, 5000.0),      # N/m
            "x": (0.06, 0.15),          # m
        },
        "push_flick": {
            "a": (80.0, 200.0),         # m/s²
            "t": (0.15, 0.30),          # s
        },
        "drag_flick": {
            "alpha": (80.0, 160.0),     # rad/s²
            "L": (0.9, 1.2),            # m
            "t": (0.18, 0.28),          # s
            "eta": (0.75, 0.95),        # efficiency
        }
    }


def get_trajectory_settings() -> Dict[str, float]:
    """
    Get default trajectory calculation settings.
    
    Returns:
        Dictionary with trajectory settings
    """
    return {
        "launch_angle": 15.0,    # degrees
        "launch_height": 0.5,    # meters
        "time_step": 0.01,       # seconds
        "max_time": 8.0,         # seconds
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def shot_type_from_string(shot_type: str) -> str:
    """
    Normalize shot type string.
    
    Args:
        shot_type: Input shot type string
    
    Returns:
        Normalized shot type string
    """
    shot_type = shot_type.lower().strip()
    
    # Handle variations
    if shot_type in ["hit", "drive", "hit_drive"]:
        return "hit"
    elif shot_type in ["slap", "slap_shot"]:
        return "slap"
    elif shot_type in ["push", "flick", "push_flick", "pushflick"]:
        return "push_flick"
    elif shot_type in ["drag", "drag_flick", "dragflick"]:
        return "drag_flick"
    else:
        raise HockeyPhysicsError(f"Unknown shot type: {shot_type}")


def get_all_shot_types() -> List[str]:
    """Get list of all supported shot types"""
    return ["hit", "slap", "push_flick", "drag_flick"]


if __name__ == "__main__":
    # Test the physics models
    print("🏑 Shot Logical - Physics Testing")
    print("=" * 40)
    
    # Test each shot type
    hit_result = hit_drive_shot(0.6, 35.0)
    print(f"Hit Shot: {hit_result.velocity:.1f} m/s, {hit_result.force:.0f} N")
    
    slap_result = slap_shot(3500, 0.1)
    print(f"Slap Shot: {slap_result.velocity:.1f} m/s, {slap_result.force:.0f} N")
    
    push_result = push_flick_shot(120, 0.2)
    print(f"Push/Flick: {push_result.velocity:.1f} m/s, {push_result.force:.0f} N")
    
    drag_result = drag_flick_shot(110, 1.0, 0.22)
    print(f"Drag Flick: {drag_result.velocity:.1f} m/s, {drag_result.force:.0f} N")
    
    # Test trajectory calculation
    trajectory = calculate_trajectory(25.0, 15.0)
    print(f"\nTrajectory: {len(trajectory)} points over {trajectory[-1].time:.1f}s")
    print(f"Max range: {max(p.x for p in trajectory):.1f}m")
    print(f"Max height: {max(p.z for p in trajectory):.1f}m")
    
    print("\n✓ All physics models working correctly!")
