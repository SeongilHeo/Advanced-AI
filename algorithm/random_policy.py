from typing import Dict, List, Optional

import numpy as np

from environment.gym import get_env

from .common import Algorithm


class RandomPolicy(Algorithm):
    def __init__(self, args):
        self.args = args
        self.env_name, self.env_kwargs = get_env(args.env_name)
        self.mode = str(getattr(args, "mode", "eval")).lower()
        self.render_mode = "human" if args.human_render else None
        self.num_eval_episodes = int(getattr(args, "num_eval_episodes", 20))
        self.num_render_episodes = int(getattr(args, "num_render_episodes", 5))
        self.seed = int(getattr(args, "seed", 0))
        self._init_artifact_paths(args)

        self.eval_returns: Dict[str, List[float]] = {"episode_rewards": []}
        self.eval_summary: Dict[str, float] = {}

    def _run_episode(self, env, seed: Optional[int] = None) -> float:
        _, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0

        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

        return total_reward

    def _evaluate_policy(self, num_episodes: int, render_mode: Optional[str] = None) -> float:
        env = self._make_env(render_mode=render_mode)
        rewards: List[float] = []

        for episode_idx in range(num_episodes):
            episode_seed = self.seed + episode_idx
            episode_reward = self._run_episode(env, seed=episode_seed)
            rewards.append(episode_reward)
            print(f"[eval] episode {episode_idx + 1}/{num_episodes} return={episode_reward:.1f}")

        env.close()

        self.eval_returns["episode_rewards"] = rewards
        if rewards:
            self.eval_summary = {
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "mean_reward": float(np.mean(rewards)),
            }
            print(
                "[Summary] "
                + f"min: {self.eval_summary['min_reward']} \t"
                + f"max: {self.eval_summary['max_reward']} \t"
                + f"mean: {self.eval_summary['mean_reward']}"
            )
            return self.eval_summary["mean_reward"]

        self.eval_summary = {"min_reward": 0.0, "max_reward": 0.0, "mean_reward": 0.0}
        return 0.0

    def _build_hparams(self):
        return dict(vars(self.args))

    def _save_eval_results(self):
        run_dir = self._configure_output_paths(create_run_dir=True)
        print(f"[Info] (random) Run directory: {run_dir}")

        self._save_training_results(
            results={
                "eval_return": self.eval_returns,
                "summary": self.eval_summary,
            },
            hparams=self._build_hparams(),
            extra_payload={
                "algorithm": "random",
                "env_name": self.env_name,
                "mode": "eval",
            },
        )
        print(f"[Info] (random) Saved results to {self.results_path}")

    def _evaluate(self):
        avg_reward = self._evaluate_policy(self.num_eval_episodes, render_mode=None)
        self._save_eval_results()
        print(
            f"[Info] (random) Average return over {self.num_eval_episodes} episodes: {avg_reward:.3f}"
        )

    def _render(self):
        avg_reward = self._evaluate_policy(
            self.num_render_episodes,
            render_mode=self.render_mode,
        )
        print(
            f"[Info] (random) Average return over {self.num_render_episodes} episodes: {avg_reward:.3f}"
        )

    def process(self):
        if self.mode == "render":
            self._render()
            print("Done.")
            return

        if self.mode == "train":
            print("[Info] (random) 'train' mode is treated as 'eval' for random policy.")
            self.mode = "eval"

        if self.mode == "eval":
            self._evaluate()
            print("Done.")
            return

        raise ValueError(f"Unknown mode: {self.mode}. Use train, eval, or render.")
