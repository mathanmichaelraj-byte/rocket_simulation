"""Integrates forces and torques to advance the rocket state."""

from __future__ import annotations
import numpy as np
from .rocket_model import RocketState
from ..config import PhysicsConfig, RocketConfig


def step(
    state: RocketState,
    thrust_n: float,
    gimbal_rad: float,
    cfg_phys: PhysicsConfig,
    cfg_rocket: RocketConfig,
    dt: float | None = None,
) -> RocketState:
    """Return a new RocketState after one timestep."""
    dt = dt or cfg_phys.timestep_s / cfg_phys.substeps
    mass = state.total_mass(cfg_rocket.dry_mass_kg)

    # Thrust components (gimbal rotates thrust vector)
    fx = thrust_n * np.sin(gimbal_rad + state.angle)
    fy = thrust_n * np.cos(gimbal_rad + state.angle)

    # Gravity
    fy -= mass * cfg_phys.gravity_mps2

    # Drag (opposing velocity)
    speed = np.hypot(state.vx, state.vy)
    if speed > 0:
        drag = 0.5 * 1.225 * speed**2 * cfg_phys.drag_coefficient * cfg_phys.reference_area_m2
        fx -= drag * (state.vx / speed)
        fy -= drag * (state.vy / speed)

    ax, ay = fx / mass, fy / mass

    # Torque from gimbal offset
    torque = -thrust_n * np.sin(gimbal_rad) * (cfg_rocket.height_m * 0.4)
    alpha = torque / cfg_rocket.moment_of_inertia

    return RocketState(
        x=state.x + state.vx * dt,
        y=max(state.y + state.vy * dt, cfg_phys.ground_y_m),
        vx=state.vx + ax * dt,
        vy=state.vy + ay * dt,
        angle=state.angle + state.omega * dt,
        omega=state.omega + alpha * dt,
        fuel_kg=state.fuel_kg,
        legs_deployed=state.legs_deployed,
    )
