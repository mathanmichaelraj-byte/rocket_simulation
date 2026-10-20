"""Generates matplotlib plots from telemetry data."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .recorder import TelemetryFrame


def plot_trajectory(frames: list[TelemetryFrame], save_path: str | Path | None = None) -> None:
    xs = [f.x for f in frames]
    ys = [f.y for f in frames]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Episode Telemetry")

    axes[0, 0].plot(xs, ys)
    axes[0, 0].set(xlabel="x (m)", ylabel="y (m)", title="Trajectory")

    steps = [f.step for f in frames]
    axes[0, 1].plot(steps, [np.hypot(f.vx, f.vy) for f in frames])
    axes[0, 1].set(xlabel="step", ylabel="speed (m/s)", title="Speed")

    axes[1, 0].plot(steps, [np.degrees(f.angle) for f in frames])
    axes[1, 0].set(xlabel="step", ylabel="angle (°)", title="Tilt")

    axes[1, 1].plot(steps, [f.fuel_kg for f in frames])
    axes[1, 1].set(xlabel="step", ylabel="fuel (kg)", title="Fuel")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
