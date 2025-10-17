"""
Shot Control System
Real-time detection and correction for hockey shot classification.
"""

from .shot_control import (
    ShotControlSystem,
    create_shot_control_system,
    DeviationAnalysis,
    CorrectionAction,
    CorrectionResult
)

__all__ = [
    'ShotControlSystem',
    'create_shot_control_system',
    'DeviationAnalysis',
    'CorrectionAction',
    'CorrectionResult'
]
