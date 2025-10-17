import numpy as np
import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from .shot_logical import TrajectoryPoint, G, AIR_DENSITY, BALL_RADIUS, DRAG_COEFFICIENT


@dataclass
class EnvironmentalConditions:
    wind_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    air_density: float = AIR_DENSITY
    temperature: float = 20.0
    humidity: float = 0.5
    pressure: float = 101325.0
    turbulence_intensity: float = 0.0

@dataclass
class SpinParameters:
    spin_rate: float = 0.0
    spin_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    spin_decay_rate: float = 0.95


class AdvancedPhysicsEngine:
    def __init__(self, conditions: EnvironmentalConditions = None):
        self.conditions = conditions or EnvironmentalConditions()
        self.magnus_coefficient = 0.25
        
    def calculate_air_density(self, temperature: float, pressure: float, humidity: float) -> float:
        dry_air_density = (pressure * 0.0289644) / (8.31432 * (temperature + 273.15))
        saturation_pressure = 610.78 * math.exp(17.27 * temperature / (temperature + 237.3))
        vapor_pressure = humidity * saturation_pressure
        return dry_air_density * (1 - 0.378 * vapor_pressure / pressure)
    
    def calculate_reynolds_number(self, velocity: float, air_density: float) -> float:
        return (air_density * velocity * 2 * BALL_RADIUS) / 1.81e-5
    
    def get_drag_coefficient(self, reynolds_number: float) -> float:
        if reynolds_number < 1e3:
            return 24 / reynolds_number
        elif reynolds_number < 1e5:
            return 0.47
        elif reynolds_number < 3e5:
            return 0.47 - 0.3 * (reynolds_number - 1e5) / (2e5)
        else:
            return 0.17
    
    def calculate_magnus_force(self, velocity: np.ndarray, spin_params: SpinParameters) -> np.ndarray:
        if spin_params.spin_rate == 0 or np.linalg.norm(velocity) == 0:
            return np.zeros(3)
        
        spin_vector = np.array(spin_params.spin_axis) * spin_params.spin_rate
        cross_product = np.cross(spin_vector, velocity)
        area = math.pi * BALL_RADIUS**2
        
        return 0.5 * self.conditions.air_density * area * self.magnus_coefficient * cross_product
    
    def calculate_wind_effect(self, velocity: np.ndarray) -> np.ndarray:
        wind = np.array(self.conditions.wind_velocity)
        relative_velocity = velocity - wind
        relative_speed = np.linalg.norm(relative_velocity)
        
        if relative_speed == 0:
            return np.zeros(3)
        
        reynolds = self.calculate_reynolds_number(relative_speed, self.conditions.air_density)
        drag_coeff = self.get_drag_coefficient(reynolds)
        area = math.pi * BALL_RADIUS**2
        drag_magnitude = 0.5 * self.conditions.air_density * area * drag_coeff * relative_speed**2
        
        return drag_magnitude * (-relative_velocity / relative_speed)
    
    def calculate_turbulence_effect(self, position: np.ndarray, time: float) -> np.ndarray:
        if self.conditions.turbulence_intensity == 0:
            return np.zeros(3)
        
        amplitude = self.conditions.turbulence_intensity * 5.0
        noise_x = amplitude * math.sin(2 * position[0] + 3 * time) * math.cos(2 * position[1])
        noise_y = amplitude * math.cos(2 * position[1] + 2 * time) * math.sin(2 * position[2])
        noise_z = amplitude * math.sin(2 * position[2] + time) * math.cos(2 * position[0])
        
        return np.array([noise_x, noise_y, noise_z])
    
    def update_spin_decay(self, spin_params: SpinParameters, dt: float) -> SpinParameters:
        new_spin_rate = spin_params.spin_rate * (spin_params.spin_decay_rate ** dt)
        return SpinParameters(new_spin_rate, spin_params.spin_axis, spin_params.spin_decay_rate)
    
    def calculate_advanced_trajectory(self, initial_velocity: float, launch_angle: float,
                                    launch_height: float = 0.5, lateral_angle: float = 0.0,
                                    spin_params: SpinParameters = None, time_step: float = 0.01,
                                    max_time: float = 10.0) -> List[TrajectoryPoint]:
        if spin_params is None:
            spin_params = SpinParameters()
        
        launch_rad = math.radians(launch_angle)
        lateral_rad = math.radians(lateral_angle)
        
        position = np.array([0.0, 0.0, launch_height])
        velocity = np.array([
            initial_velocity * math.cos(launch_rad) * math.cos(lateral_rad),
            initial_velocity * math.cos(launch_rad) * math.sin(lateral_rad),
            initial_velocity * math.sin(launch_rad)
        ])
        
        trajectory = []
        time = 0.0
        current_spin = spin_params
        
        self.conditions.air_density = self.calculate_air_density(
            self.conditions.temperature, self.conditions.pressure, self.conditions.humidity
        )
        
        while time <= max_time and position[2] >= 0:
            trajectory.append(TrajectoryPoint(time, position[0], position[1], position[2]))
            
            gravity_force = np.array([0.0, 0.0, -G])
            magnus_force = self.calculate_magnus_force(velocity, current_spin)
            wind_drag = self.calculate_wind_effect(velocity)
            turbulence_force = self.calculate_turbulence_effect(position, time)
            
            total_acceleration = gravity_force + magnus_force + wind_drag + turbulence_force
            
            new_velocity = velocity + total_acceleration * time_step
            new_position = position + velocity * time_step + 0.5 * total_acceleration * time_step**2
            
            position = new_position
            velocity = new_velocity
            time += time_step
            current_spin = self.update_spin_decay(current_spin, time_step)
        
        return trajectory


class EffectsPresets:
    @staticmethod
    def calm_conditions() -> EnvironmentalConditions:
        return EnvironmentalConditions()
    
    @staticmethod
    def windy_conditions() -> EnvironmentalConditions:
        return EnvironmentalConditions(wind_velocity=(3.0, 1.5, 0.0), turbulence_intensity=0.3)
    
    @staticmethod
    def humid_conditions() -> EnvironmentalConditions:
        return EnvironmentalConditions(temperature=30.0, humidity=0.9, pressure=101000.0)
    
    @staticmethod
    def high_altitude() -> EnvironmentalConditions:
        return EnvironmentalConditions(temperature=10.0, pressure=85000.0, humidity=0.3)
    
    @staticmethod
    def indoor_conditions() -> EnvironmentalConditions:
        return EnvironmentalConditions(temperature=22.0, humidity=0.4, turbulence_intensity=0.05)


def simulate_shot_with_effects(shot_type: str, base_velocity: float,
                             conditions: EnvironmentalConditions = None,
                             spin_params: SpinParameters = None) -> List[TrajectoryPoint]:
    if conditions is None:
        conditions = EffectsPresets.calm_conditions()
    
    if spin_params is None:
        spin_configs = {
            'hit': SpinParameters(15.0, (0.2, 0.1, 0.97)),
            'slap': SpinParameters(25.0, (0.1, 0.0, 0.99)),
            'push_flick': SpinParameters(10.0, (0.0, 0.3, 0.95)),
            'drag_flick': SpinParameters(35.0, (0.3, 0.2, 0.93))
        }
        spin_params = spin_configs.get(shot_type, SpinParameters())
    
    launch_angles = {'hit': 12.0, 'slap': 15.0, 'push_flick': 18.0, 'drag_flick': 22.0}
    launch_angle = launch_angles.get(shot_type, 15.0)
    
    engine = AdvancedPhysicsEngine(conditions)
    return engine.calculate_advanced_trajectory(base_velocity, launch_angle, spin_params=spin_params)
