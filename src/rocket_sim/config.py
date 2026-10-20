"""Loads and merges all YAML configs into typed dataclasses."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


CONFIG_DIR = Path(__file__).parents[3] / "configs"


def _load(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


@dataclass
class PhysicsConfig:
    gravity_mps2: float = 9.80665
    timestep_s: float = 0.05
    substeps: int = 4
    drag_coefficient: float = 0.3
    reference_area_m2: float = 10.75
    ground_y_m: float = 0.0
    max_landing_velocity_mps: float = 2.0
    max_landing_angle_deg: float = 5.0


@dataclass
class RocketConfig:
    dry_mass_kg: float = 25000.0
    fuel_mass_kg: float = 10000.0
    height_m: float = 50.0
    radius_m: float = 1.85
    moment_of_inertia: float = 1.2e6
    max_gimbal_angle_deg: float = 5.0


@dataclass
class TrainingConfig:
    algorithm: str = "PPO"
    total_timesteps: int = 5_000_000
    n_envs: int = 8
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    log_dir: str = "logs/"
    model_dir: str = "models/"


@dataclass
class EnvironmentConfig:
    initial_altitude_m: float = 1500.0
    pad_x_m: float = 0.0
    pad_half_width_m: float = 5.0
    max_episode_steps: int = 1000


def load_physics() -> PhysicsConfig:
    data = _load("physics.yaml")["physics"]
    return PhysicsConfig(**{k: v for k, v in data.items() if hasattr(PhysicsConfig, k)})


def load_rocket() -> RocketConfig:
    data = _load("rocket.yaml")["rocket"]
    return RocketConfig(**{k: v for k, v in data.items() if hasattr(RocketConfig, k)})


def load_training() -> TrainingConfig:
    data = _load("training.yaml")["training"]
    return TrainingConfig(**{k: v for k, v in data.items() if hasattr(TrainingConfig, k)})


def load_environment() -> EnvironmentConfig:
    data = _load("environment.yaml")["environment"]
    return EnvironmentConfig(**{k: v for k, v in data.items() if hasattr(EnvironmentConfig, k)})
