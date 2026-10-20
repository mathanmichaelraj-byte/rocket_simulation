"""Landing leg deployment and ground contact logic."""

from __future__ import annotations
from .rocket_model import RocketState


DEPLOY_ALTITUDE_M = 500.0
LEG_SPAN_M = 6.0


def maybe_deploy(state: RocketState) -> RocketState:
    """Auto-deploy legs below the deployment altitude."""
    if not state.legs_deployed and state.y <= DEPLOY_ALTITUDE_M:
        return RocketState(**{**state.__dict__, "legs_deployed": True})
    return state


def contact_points(state: RocketState) -> list[tuple[float, float]]:
    """Return (x, y) of each leg tip (simplified 2-leg model)."""
    import numpy as np
    half = LEG_SPAN_M / 2
    cos_a, sin_a = np.cos(state.angle), np.sin(state.angle)
    return [
        (state.x - half * cos_a, state.y - half * sin_a),
        (state.x + half * cos_a, state.y + half * sin_a),
    ]
