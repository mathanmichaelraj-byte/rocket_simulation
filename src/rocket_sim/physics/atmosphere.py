"""Simple ISA atmosphere model — density vs altitude."""

from __future__ import annotations
import numpy as np

SEA_LEVEL_DENSITY = 1.225   # kg/m³
SCALE_HEIGHT = 8500.0       # m  (exponential approximation)


def air_density(altitude_m: float) -> float:
    """Return air density (kg/m³) at the given altitude using an exponential model."""
    return SEA_LEVEL_DENSITY * np.exp(-altitude_m / SCALE_HEIGHT)
