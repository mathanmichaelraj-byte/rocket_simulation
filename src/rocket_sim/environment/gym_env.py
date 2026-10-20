"""Gymnasium environment wrapping the rocket physics simulation."""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..config import load_physics, load_rocket, load_environment
from ..physics.rocket_model import RocketState
from ..physics.dynamics import step as physics_step
from ..physics.propulsion import compute_thrust, fuel_flow_rate
from ..physics.wind import WindModel
from ..physics.landing_gear import maybe_deploy
from .observations import build_observation, OBS_DIM
from .actions import decode_action, ACTION_DIM
from .rewards import compute_reward
from .termination import check_termination


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.phys = load_physics()
        self.rocket = load_rocket()
        self.env_cfg = load_environment()
        self._wind = WindModel()

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self._state: RocketState | None = None
        self._prev_state: RocketState | None = None
        self._step_count = 0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self._state = RocketState(
            x=float(rng.uniform(-200, 200)),
            y=self.env_cfg.initial_altitude_m,
            vx=float(rng.uniform(-20, 20)),
            vy=float(rng.uniform(-80, -40)),
            angle=float(rng.uniform(-0.17, 0.17)),
            omega=float(rng.uniform(-0.1, 0.1)),
            fuel_kg=self.rocket.fuel_mass_kg,
        )
        self._prev_state = self._state
        self._step_count = 0
        self._wind.reset()
        return build_observation(self._state, self.env_cfg.pad_x_m, self.rocket.fuel_mass_kg), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        import math
        cmd = decode_action(action, math.radians(self.rocket.max_gimbal_angle_deg))
        thrust = compute_thrust(cmd.throttle) if self._state.fuel_kg > 0 else 0.0
        fuel_used = fuel_flow_rate(thrust) * self.phys.timestep_s

        self._prev_state = self._state
        new_state = self._state

        # Sub-step integration
        sub_dt = self.phys.timestep_s / self.phys.substeps
        for _ in range(self.phys.substeps):
            new_state = physics_step(new_state, thrust, cmd.gimbal, self.phys, self.rocket, sub_dt)

        new_state.fuel_kg = max(0.0, self._state.fuel_kg - fuel_used)
        new_state = maybe_deploy(new_state)
        self._state = new_state
        self._step_count += 1

        done, landed, crashed, oob = check_termination(
            self._state, self.env_cfg.pad_x_m, self.phys,
            self._step_count, self.env_cfg.max_episode_steps,
        )
        reward = compute_reward(
            self._state, self._prev_state, self.env_cfg.pad_x_m,
            self.phys, cmd.throttle, landed, crashed, oob,
        )
        obs = build_observation(self._state, self.env_cfg.pad_x_m, self.rocket.fuel_mass_kg)
        info = {"landed": landed, "crashed": crashed, "out_of_bounds": oob}
        return obs, reward, done, False, info
