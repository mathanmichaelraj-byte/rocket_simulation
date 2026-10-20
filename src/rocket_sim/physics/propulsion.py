"""Main engine and RCS thrust calculations."""

from __future__ import annotations
import numpy as np


# Merlin-1D vacuum-ish constants (simplified)
MAIN_ENGINE_MAX_THRUST_N = 845_000.0
MAIN_ENGINE_MIN_THROTTLE = 0.40       # can't throttle below 40 %
SPECIFIC_IMPULSE_S = 311.0            # Isp in seconds
G0 = 9.80665


def compute_thrust(throttle: float) -> float:
    """Return thrust (N) for a throttle in [0, 1]."""
    if throttle <= 0:
        return 0.0
    effective = MAIN_ENGINE_MIN_THROTTLE + throttle * (1.0 - MAIN_ENGINE_MIN_THROTTLE)
    return effective * MAIN_ENGINE_MAX_THRUST_N


def fuel_flow_rate(thrust_n: float) -> float:
    """Return fuel consumption rate (kg/s)."""
    return thrust_n / (SPECIFIC_IMPULSE_S * G0)
