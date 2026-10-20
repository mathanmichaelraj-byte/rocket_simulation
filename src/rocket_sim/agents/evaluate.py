"""Run a trained model and report landing statistics."""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from ..environment.gym_env import RocketLandingEnv


def evaluate(model_path: str, n_episodes: int = 20) -> None:
    env = RocketLandingEnv()
    model = PPO.load(model_path)

    results = {"landed": 0, "crashed": 0, "oob": 0}
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, total_r = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(action)
            total_r += r
        rewards.append(total_r)
        if info["landed"]:
            results["landed"] += 1
        elif info["crashed"]:
            results["crashed"] += 1
        else:
            results["oob"] += 1

    print(f"Episodes : {n_episodes}")
    print(f"Landed   : {results['landed']} ({results['landed']/n_episodes*100:.1f}%)")
    print(f"Crashed  : {results['crashed']}")
    print(f"OOB      : {results['oob']}")
    print(f"Avg reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
