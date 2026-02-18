from .common import Algorithm

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from environment.gym import get_env

Action = int
State = Tuple[int, int, bool]
QTable = Dict[State, np.ndarray]


class Q(Algorithm):
    def __init__(self, args):
        self.args=args
        self.device = args.device
        self.env_name, self.env_kwargs = get_env(args.env_name)
        self.render_mode = "human" if args.human_render else None
        self.mode = str(getattr(args, "mode", "train")).lower()

        self.num_episodes = int(getattr(args, "num_episodes", 100_000))
        self.lr = float(args.lr)
        self.discount_factor = float(args.discount_factor)

        self.exploration_mode = str(args.exploration_mode)
        if self.exploration_mode not in {"epsilon_greedy", "boltzmann"}:
            raise ValueError(f"Unknown exploration_mode: {self.exploration_mode}")

        self.epsilon_start = float(args.epsilon_start)
        self.epsilon_final = float(args.epsilon_final)
        self.epsilon_decay = float(args.epsilon_decay)
        self.epsilon_schedule = str(args.epsilon_schedule)
        self.epsilon = float(args.epsilon)
        self.temperature = float(args.temperature)

        self.eval_interval = int(args.eval_interval)
        self.num_eval_episodes = int(getattr(args, "num_eval_episodes", 500))
        self.render_episodes = int(getattr(args, "num_render_episodes", 5))
        self.seed = int(args.seed)

        self._init_artifact_paths(args)

        self.rng = np.random.default_rng(self.seed)

        # Infer action-space size once and build table defaults.
        warmup_env = self._make_env()
        self.n_actions = warmup_env.action_space.n
        warmup_env.close()
        self.q_table: QTable = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

        self._normalize_exploration_args()
        self.train_returns: Dict[str, List[float]] = {"episode_rewards": []}
        self.eval_returns: Dict[str, List[float]] = {"points": [], "episode_rewards": []}

    def _normalize_exploration_args(self):
        if self.exploration_mode == "boltzmann":
            if self.temperature <= 0:
                raise ValueError("temperature must be > 0 for boltzmann exploration")
            self.epsilon = None
            self.epsilon_schedule = None
            self.epsilon_start = None
            self.epsilon_final = None
            self.epsilon_decay = None
            return

        if self.epsilon_schedule not in {"linear", "exp", "fixed"}:
            raise ValueError(f"Unknown epsilon_schedule: {self.epsilon_schedule}")
        
        self.temperature = None
        
        if self.epsilon_schedule == "fixed":
            self.epsilon_start = None
            self.epsilon_final = None
            self.epsilon_decay = None
            return

        self.epsilon = self.epsilon_start
        if self.epsilon_schedule == "linear":
            if self.epsilon_decay <= 1:
                raise ValueError("For linear schedule, epsilon_decay must be > 1")
            self.epsilon_decay = (self.epsilon_start - self.epsilon_final) / self.epsilon_decay
            return

        if self.epsilon_decay >= 1:
            raise ValueError("For exp schedule, epsilon_decay must be < 1")

    def _decay_epsilon(self):
        if self.exploration_mode != "epsilon_greedy":
            return
        if self.epsilon_schedule == "fixed":
            return
        if self.epsilon_schedule == "linear":
            self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_decay)
            return
        if self.epsilon_schedule == "exp":
            self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)
            return
        raise ValueError(f"Unknown epsilon_schedule: {self.epsilon_schedule}")

    def _select_action(self, env, state: State, training: bool = True) -> Action:
        q_values = self.q_table[state]
        greedy = int(np.argmax(q_values))

        if not training:
            return greedy

        if self.exploration_mode == "epsilon_greedy":
            if self.rng.random() < self.epsilon:
                return int(env.action_space.sample())
            return greedy

        centered = (q_values - np.max(q_values)) / self.temperature
        probs = np.exp(centered)
        probs = probs / np.sum(probs)
        return int(self.rng.choice(self.n_actions, p=probs))

    def _update(self, state: State, action: Action, reward: float, next_state: State, done: bool):
        future_q_value = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_error = reward + self.discount_factor * future_q_value - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def _run_episode(self, env, training: bool = True) -> float:
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = self._select_action(env, state, training=training)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if training:
                self._update(state, action, float(reward), next_state, done)

            state = next_state
            total_reward += float(reward)

        if training and self.exploration_mode == "epsilon_greedy":
            self._decay_epsilon()

        return total_reward

    def _run_greedy_episode(self, env, seed: Optional[int] = None) -> float:
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0

        while not done:
            action = int(np.argmax(self.q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

        return total_reward

    def _evaluate_greedy_policy(self, env, num_episodes: int, seed: Optional[int] = None) -> float:
        rewards: List[float] = []
        for episode_idx in range(num_episodes):
            episode_seed = None if seed is None else seed + episode_idx
            rewards.append(self._run_greedy_episode(env, seed=episode_seed))
            # print(f"[Eval] ({episode_idx}) total reward: {rewards[-1]}")

        print("[Summary] "+
            f"min: {np.min(rewards)} \t"+
            f"max: {np.max(rewards)} \t" +
            f"mean: {np.mean(rewards)}",
        )
        return float(np.mean(rewards)) if rewards else 0.0

    def _build_hparams(self) -> Dict:
        hparams = {
            "episodes": self.num_episodes,
            "lr": self.lr,
            "discount_factor": self.discount_factor,
            "exploration_mode": self.exploration_mode,
            "epsilon_start": self.epsilon_start,
            "epsilon_final": self.epsilon_final,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_schedule": self.epsilon_schedule,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.num_eval_episodes,
            "seed": self.seed,
        }
        return {k: v for k, v in hparams.items() if v is not None}

    def _build_model_state(self) -> Dict:
        return {
            "n_actions": self.n_actions,
            "q_table": {key: value.astype(float) for key, value in self.q_table.items()},
        }

    def _load_model_state(self, model_state: Dict):
        loaded_n_actions = int(model_state.get("n_actions", self.n_actions))
        if loaded_n_actions != self.n_actions:
            raise ValueError("Loaded model action-space size does not match environment.")

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))
        for key, value in model_state["q_table"].items():
            self.q_table[tuple(key)] = np.array(value, dtype=float)

    def _load_checkpoint(self):
        payload = self._load_model_checkpoint()
        if "model_state" in payload:
            self._load_model_state(payload["model_state"])
            return payload

        # Backward compatibility for older q.py checkpoint shape.
        if "q_table" in payload:
            self._load_model_state(payload)
            return payload

        raise ValueError("Unsupported checkpoint format: missing model_state/q_table.")

    def _train(self):
        env = self._make_env()
        eval_env = self._make_env()
        env.reset(seed=self.seed)
        env.action_space.seed(self.seed)

        for episode in range(1, self.num_episodes + 1):
            reward = self._run_episode(env, training=True)
            self.train_returns["episode_rewards"].append(reward)

            if episode % self.eval_interval == 0:
                avg_reward = self._evaluate_greedy_policy(
                    eval_env,
                    self.num_eval_episodes,
                    seed=self.seed + episode,
                )
                self.eval_returns["points"].append(episode)
                self.eval_returns["episode_rewards"].append(avg_reward)
        
        print("[Info] (q) Save artifacts.")
        self._save_artifacts()

        env.close()
        eval_env.close()

    def _evaluate(self):
        payload = self._load_checkpoint()
        env = self._make_env()
        avg_reward = self._evaluate_greedy_policy(env, self.num_eval_episodes, seed=self.seed)
        env.close()
        print(f"[Info] (q) Loaded checkpoint: {self.checkpoint_path}")
        print(f"[Info] (q) Average return over {self.num_eval_episodes} episodes: {avg_reward:.3f}")
        if "hparams" in payload:
            print(f"[Info] (q) Loaded hparams keys: {sorted(payload['hparams'].keys())}")

    def _render(self):
        print("[Info] (q) Load checkpoint.")
        self._load_checkpoint()
        env = self._make_env(render_mode=self.render_mode)
        for episode_idx in range(self.render_episodes):
            episode_return = self._run_greedy_episode(env, seed=self.seed + episode_idx)
            print(
                f"[eval] episode {episode_idx + 1}/{self.render_episodes} "
                f"return={episode_return:.1f}"
            )
        env.close()

    def process(self):
        if self.mode == "eval":
            self._evaluate()
            print("Done.")
            return

        if self.mode == "render":
            self._render()
            print("Done.")
            return

        if self.mode != "train":
            raise ValueError(f"Unknown mode: {self.mode}. Use train, eval, or render.")

        self._train()
        print("Done.")
