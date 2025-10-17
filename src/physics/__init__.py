"""
Physics Engine Module

Contains core physics models, shot calculations, and trajectory generation.
"""

from .shot_logical import (
    hit_drive_shot, slap_shot, push_flick_shot, drag_flick_shot,
    calculate_trajectory, ShotResult, TrajectoryPoint,
    get_parameter_ranges, get_trajectory_settings, get_all_shot_types,
    shot_type_from_string, apply_stick_flex_enhancement, calculate_environmental_effects
)

from .procedural_shot_generation import ShotGenerator

__all__ = [
    'hit_drive_shot', 'slap_shot', 'push_flick_shot', 'drag_flick_shot',
    'calculate_trajectory', 'ShotResult', 'TrajectoryPoint',
    'get_parameter_ranges', 'get_trajectory_settings', 'get_all_shot_types',
    'shot_type_from_string', 'apply_stick_flex_enhancement', 'calculate_environmental_effects',
    'ShotGenerator'
]
