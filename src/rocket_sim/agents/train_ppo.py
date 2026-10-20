"""Train a PPO agent on the RocketLandingEnv."""

from __future__ import annotations
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from ..config import load_training
from ..environment.gym_env import RocketLandingEnv


def train(args: argparse.Namespace) -> None:
    cfg = load_training()
    log_dir = Path(cfg.log_dir)
    model_dir = Path(cfg.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(RocketLandingEnv, n_envs=cfg.n_envs)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        tensorboard_log=str(log_dir),
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=cfg.checkpoint_freq // cfg.n_envs,
        save_path=str(model_dir),
        name_prefix="ppo_rocket",
    )

    model.learn(total_timesteps=cfg.total_timesteps, callback=checkpoint_cb)
    model.save(str(model_dir / "best_model"))
    print(f"Model saved to {model_dir / 'best_model'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    train(parser.parse_args())
