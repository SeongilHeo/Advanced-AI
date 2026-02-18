from .common import Algorithm

import math
import random
from collections import deque, namedtuple
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.gym import get_env
from model.cnn import CNN
from model.fcnet import FCNet


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=int(capacity))

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, int(batch_size))

    def __len__(self):
        return len(self.memory)


class DiscretizeCarRacing(gym.ActionWrapper):
    """Discretize CarRacing continuous actions into a small discrete set."""

    def __init__(self, env):
        super().__init__(env)
        self._actions = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),   # no-op
            np.array([0.0, 1.0, 0.0], dtype=np.float32),   # gas
            np.array([0.0, 0.0, 0.8], dtype=np.float32),   # brake
            np.array([-1.0, 0.5, 0.0], dtype=np.float32),  # left + gas
            np.array([1.0, 0.5, 0.0], dtype=np.float32),   # right + gas
        ]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, act):
        return self._actions[int(act)]


class DQN(Algorithm):
    def __init__(self, args):
        self.args=args
        self.env_name, self.env_kwargs = get_env(args.env_name)
        self.render_mode = "human" if args.human_render else None
        self.mode = str(getattr(args, "mode", "train")).lower()
        self._init_artifact_paths(args)

        self.num_episodes = int(getattr(args, "num_episodes", 500))
        self.batch_size = int(args.batch_size)
        self.gamma = float(args.gamma)
        self.eps_start = float(args.eps_start)
        self.eps_end = float(args.eps_end)
        self.eps_decay = float(args.eps_decay)
        self.tau = float(args.tau)
        self.lr = float(args.lr)
        self.memory_capacity = int(args.memory_capacity)
        self.eval_interval = int(getattr(args, "eval_interval", 50))
        self.num_eval_episodes = int(getattr(args, "num_eval_episodes", 20))
        self.num_render_episodes = int(getattr(args, "num_render_episodes", 5))
        self.log_interval = int(getattr(args, "log_interval", 10))
        self.seed = int(args.seed)
        self.network_type = str(getattr(args, "network_type", "fcn")).lower()
        if self.network_type not in {"fcn", "cnn"}:
            raise ValueError(
                f"Unknown network_type: {self.network_type}. Use 'fcn' or 'cnn'."
            )
        self.cnn_image_size = int(getattr(args, "cnn_image_size", 84))
        if self.cnn_image_size <= 0:
            raise ValueError("cnn_image_size must be > 0.")

        self.device = self._resolve_device(args.device)
        self._seed_everything(self.seed)
        self.n_observations, self.n_actions, self.obs_shape = self._infer_space_sizes()

        self.hidden_dim = int(args.hidden_dim)
        self.num_layers = int(args.num_layers)
        self.activation = args.activation

        self.cnn_head_hidden_dim = int(getattr(args, "cnn_head_hidden_dim", 512))
        self.cnn_head_num_layers = int(getattr(args, "cnn_head_num_layers", 0))
        self.cnn_conv_activation = str(getattr(args, "cnn_conv_activation", "relu"))
        self.cnn_head_activation = str(getattr(args, "cnn_head_activation", "relu"))

        self.policy_net, self.target_net = self._build_networks()
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(self.memory_capacity)

        self.steps_done = 0
        self.train_returns: Dict[str, List[float]] = {"episode_rewards": [], "losses": []}
        self.eval_returns: Dict[str, List[float]] = {"points": [], "episode_rewards": []}

    def _make_env(self, render_mode=None, **make_kwargs):
        effective_render_mode = render_mode
        if effective_render_mode is None and self.network_type == "cnn":
            # For image-based DQN, keep rgb_array available for render() fallback.
            effective_render_mode = "rgb_array"

        env = super()._make_env(render_mode=effective_render_mode, **make_kwargs)
        if self.env_name == "CarRacing-v3":
            env = DiscretizeCarRacing(env)
        return env

    def _infer_space_sizes(self):
        env = self._make_env()
        try:
            if not hasattr(env.action_space, "n"):
                raise ValueError(
                    f"DQN requires a discrete action space, but got {type(env.action_space).__name__}"
                )
            state, _ = env.reset()
            if self.network_type == "cnn":
                side = self.cnn_image_size
                n_observations = int(side * side)
                obs_shape = (1, side, side)
            else:
                state_np = np.asarray(state, dtype=np.float32)
                n_observations = int(state_np.reshape(-1).shape[0])
                obs_shape = tuple(state_np.shape)
            n_actions = int(env.action_space.n)
            return n_observations, n_actions, obs_shape
        finally:
            env.close()

    def _infer_cnn_input_spec(self):
        if len(self.obs_shape) == 2:
            in_channels = 1
            input_hw = (int(self.obs_shape[0]), int(self.obs_shape[1]))
            obs_layout = "hw"
            return in_channels, input_hw, obs_layout

        if len(self.obs_shape) == 3:
            # Prefer CHW if first dim looks like a channel count.
            if int(self.obs_shape[0]) in {1, 3, 4}:
                in_channels = int(self.obs_shape[0])
                input_hw = (int(self.obs_shape[1]), int(self.obs_shape[2]))
                obs_layout = "chw"
            else:
                in_channels = int(self.obs_shape[2])
                input_hw = (int(self.obs_shape[0]), int(self.obs_shape[1]))
                obs_layout = "hwc"
            return in_channels, input_hw, obs_layout

        raise ValueError(
            "CNN network_type requires 2D/3D observations, "
            f"but got shape {self.obs_shape}."
        )

    def _build_networks(self):
        if self.network_type == "fcn":
            self.obs_layout = "flat"
            policy_net = FCNet(
                hidden_dim=self.hidden_dim,
                input_dim=self.n_observations,
                output_dim=self.n_actions,
                num_layers=self.num_layers,
                activation=self.activation,
            )
            target_net = FCNet(
                hidden_dim=self.hidden_dim,
                input_dim=self.n_observations,
                output_dim=self.n_actions,
                num_layers=self.num_layers,
                activation=self.activation,
            )
            return policy_net, target_net

        in_channels, input_hw, obs_layout = self._infer_cnn_input_spec()
        self.obs_layout = obs_layout
        policy_net = CNN(
            in_channels=in_channels,
            n_actions=self.n_actions,
            input_hw=input_hw,
            head_hidden_dim=self.cnn_head_hidden_dim,
            head_num_layers=self.cnn_head_num_layers,
            conv_activation=self.cnn_conv_activation,
            head_activation=self.cnn_head_activation,
        )
        target_net = CNN(
            in_channels=in_channels,
            n_actions=self.n_actions,
            input_hw=input_hw,
            head_hidden_dim=self.cnn_head_hidden_dim,
            head_num_layers=self.cnn_head_num_layers,
            conv_activation=self.cnn_conv_activation,
            head_activation=self.cnn_head_activation,
        )
        return policy_net, target_net

    def _epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )

    def _render_frame(self, env) -> np.ndarray:
        frame = env.render()
        if frame is None:
            raise RuntimeError("render() returned None. Ensure env uses render_mode='rgb_array'.")
        return np.asarray(frame)

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame_np = np.asarray(frame)
        if frame_np.ndim != 3:
            raise ValueError(f"Expected image frame with 3 dims, got shape {frame_np.shape}.")

        # If CHW is provided, convert to HWC.
        if frame_np.shape[0] in {1, 3, 4} and frame_np.shape[-1] not in {1, 3, 4}:
            frame_np = np.transpose(frame_np, (1, 2, 0))

        if frame_np.shape[2] == 1:
            gray = frame_np[..., 0]
        else:
            gray = (
                frame_np[..., 0] * 0.2989
                + frame_np[..., 1] * 0.5870
                + frame_np[..., 2] * 0.1140
            )
        gray = gray.astype(np.float32)
        if gray.size > 0 and float(np.max(gray)) > 1.0:
            gray = gray / 255.0

        side = self.cnn_image_size
        h, w = gray.shape
        ys = np.linspace(0, h - 1, side).astype(np.int32)
        xs = np.linspace(0, w - 1, side).astype(np.int32)
        resized = gray[ys][:, xs]

        return torch.tensor(resized, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

    def _obs_to_tensor(self, obs: Any, env=None) -> torch.Tensor:
        if self.network_type == "fcn":
            obs_np = np.asarray(obs, dtype=np.float32)
            obs_np = obs_np.reshape(1, -1)
            return torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        # CNN path: lunar_lander_model.py style preprocessing.
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            frame = obs
        else:
            if env is None:
                raise ValueError("CNN preprocessing requires env for render() fallback.")
            frame = self._render_frame(env)
        return self._preprocess_frame(frame)

    def _select_action(self, env, state_t: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            with torch.no_grad():
                return self.policy_net(state_t).max(1).indices.view(1, 1)

        sample = random.random()
        eps_threshold = self._epsilon()
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state_t).max(1).indices.view(1, 1)
        return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)

    def _optimize_model(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(next_state is not None for next_state in batch.next_state),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = [next_state for next_state in batch.next_state if next_state is not None]

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(torch.cat(non_final_next_states)).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        target_sd = self.target_net.state_dict()
        policy_sd = self.policy_net.state_dict()
        for key in policy_sd:
            target_sd[key] = policy_sd[key] * self.tau + target_sd[key] * (1.0 - self.tau)
        self.target_net.load_state_dict(target_sd)

        return float(loss.item())

    def _run_train_episode(self, env) -> float:
        state, _ = env.reset()
        state_t = self._obs_to_tensor(state, env=env)
        episode_return = 0.0

        done = False
        while not done:
            action = self._select_action(env, state_t, training=True)
            obs2, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            reward_t = torch.tensor([float(reward)], device=self.device)
            episode_return += float(reward)

            if done:
                next_state_t = None
            else:
                next_state_t = self._obs_to_tensor(obs2, env=env)

            self.memory.push(state_t, action, next_state_t, reward_t)
            if next_state_t is not None:
                state_t = next_state_t

            loss = self._optimize_model()
            if loss is not None:
                self.train_returns["losses"].append(loss)

        return episode_return

    def _run_greedy_episode(self, env, seed: Optional[int] = None) -> float:
        state, _ = env.reset(seed=seed)
        state_t = self._obs_to_tensor(state, env=env)
        done = False
        total_reward = 0.0

        while not done:
            action = self._select_action(env, state_t, training=False)
            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            total_reward += float(reward)
            if not done:
                state_t = self._obs_to_tensor(next_state, env=env)

        return total_reward

    def _evaluate_policy(self, env, num_episodes: int, seed: Optional[int] = None) -> float:
        rewards: List[float] = []
        for episode_idx in range(num_episodes):
            episode_seed = None if seed is None else seed + episode_idx
            rewards.append(self._run_greedy_episode(env, seed=episode_seed))

        if rewards:
            print(
                "[Summary] "
                + f"min: {np.min(rewards)} \t"
                + f"max: {np.max(rewards)} \t"
                + f"mean: {np.mean(rewards)}"
            )
        return float(np.mean(rewards)) if rewards else 0.0

    def _build_hparams(self) -> Dict[str, Any]:
        hparams = {
            "num_episodes": self.num_episodes,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay": self.eps_decay,
            "tau": self.tau,
            "lr": self.lr,
            "memory_capacity": self.memory_capacity,
            "eval_interval": self.eval_interval,
            "num_eval_episodes": self.num_eval_episodes,
            "num_render_episodes": self.num_render_episodes,
            "log_interval": self.log_interval,
            "seed": self.seed,
            "device": str(self.device),
            "env_name": self.env_name,
            "network_type": self.network_type,
            "cnn_image_size": self.cnn_image_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "activation": self.activation,
            "cnn_head_hidden_dim": self.cnn_head_hidden_dim,
            "cnn_head_num_layers": self.cnn_head_num_layers,
            "cnn_conv_activation": self.cnn_conv_activation,
            "cnn_head_activation": self.cnn_head_activation,
        }
        return {k: v for k, v in hparams.items() if v is not None}

    def _build_model_state(self) -> Dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "n_actions": self.n_actions,
            "obs_shape": self.obs_shape,
            "obs_layout": self.obs_layout,
            "network_type": self.network_type,
            "cnn_image_size": self.cnn_image_size,
            "steps_done": self.steps_done,
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def _load_model_state(self, model_state: Dict[str, Any]):
        loaded_n_obs = int(model_state.get("n_observations", self.n_observations))
        loaded_n_actions = int(model_state.get("n_actions", self.n_actions))
        if loaded_n_obs != self.n_observations:
            raise ValueError("Loaded model observation size does not match environment.")
        if loaded_n_actions != self.n_actions:
            raise ValueError("Loaded model action-space size does not match environment.")
        loaded_network_type = str(model_state.get("network_type", self.network_type))
        if loaded_network_type != self.network_type:
            raise ValueError(
                "Loaded model network_type does not match current configuration: "
                f"{loaded_network_type} != {self.network_type}"
            )
        if self.network_type == "cnn":
            loaded_cnn_image_size = int(model_state.get("cnn_image_size", self.cnn_image_size))
            if loaded_cnn_image_size != self.cnn_image_size:
                raise ValueError(
                    "Loaded model cnn_image_size does not match current configuration: "
                    f"{loaded_cnn_image_size} != {self.cnn_image_size}"
                )
            loaded_obs_shape = tuple(model_state.get("obs_shape", self.obs_shape))
            if loaded_obs_shape != tuple(self.obs_shape):
                raise ValueError(
                    "Loaded model observation shape does not match environment: "
                    f"{loaded_obs_shape} != {self.obs_shape}"
                )

        if "policy_state_dict" not in model_state:
            raise ValueError("Checkpoint missing policy_state_dict.")

        self.policy_net.load_state_dict(model_state["policy_state_dict"])
        target_sd = model_state.get("target_state_dict", model_state["policy_state_dict"])
        self.target_net.load_state_dict(target_sd)

        if model_state.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(model_state["optimizer_state_dict"])

        self.steps_done = int(model_state.get("steps_done", 0))
        self.policy_net.eval()
        self.target_net.eval()

    def _load_checkpoint(self) -> Dict[str, Any]:
        payload = self._load_model_checkpoint()
        if "model_state" in payload:
            self._load_model_state(payload["model_state"])
            return payload

        if "policy_state_dict" in payload:
            self._load_model_state(payload)
            return payload

        raise ValueError("Unsupported checkpoint format: missing model_state/policy_state_dict.")

    def _train(self):
        env = self._make_env()
        eval_env = self._make_env()
        env.reset(seed=self.seed)
        env.action_space.seed(self.seed)

        self.policy_net.train()
        self.target_net.train()

        for episode in range(1, self.num_episodes + 1):
            episode_reward = self._run_train_episode(env)
            self.train_returns["episode_rewards"].append(episode_reward)

            if self.log_interval > 0 and (
                episode % self.log_interval == 0 or episode == self.num_episodes
            ):
                recent_rewards = self.train_returns["episode_rewards"][-self.log_interval :]
                avg_return = float(np.mean(recent_rewards)) if recent_rewards else float("nan")
                recent_losses = self.train_returns["losses"][-200:]
                avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
                print(
                    f"[train] episode {episode}/{self.num_episodes} "
                    f"avg_return={avg_return:.2f} avg_loss={avg_loss:.4f} "
                    f"epsilon={self._epsilon():.3f} steps={self.steps_done}"
                )

            if self.eval_interval > 0 and episode % self.eval_interval == 0:
                self.policy_net.eval()
                avg_reward = self._evaluate_policy(
                    eval_env,
                    self.num_eval_episodes,
                    seed=self.seed + episode,
                )
                self.policy_net.train()
                self.eval_returns["points"].append(episode)
                self.eval_returns["episode_rewards"].append(avg_reward)

        print("[Info] (dqn) Save artifacts.")
        self._save_artifacts()

        env.close()
        eval_env.close()

    def _evaluate(self):
        payload = self._load_checkpoint()
        env = self._make_env()
        avg_reward = self._evaluate_policy(env, self.num_eval_episodes, seed=self.seed)
        env.close()

        print(f"[Info] (dqn) Loaded checkpoint: {self.checkpoint_path}")
        print(f"[Info] (dqn) Average return over {self.num_eval_episodes} episodes: {avg_reward:.3f}")
        if "hparams" in payload:
            print(f"[Info] (dqn) Loaded hparams keys: {sorted(payload['hparams'].keys())}")

    def _render(self):
        print("[Info] (dqn) Load checkpoint.")
        self._load_checkpoint()
        env = self._make_env(render_mode=self.render_mode)
        for episode_idx in range(self.num_render_episodes):
            episode_return = self._run_greedy_episode(env, seed=self.seed + episode_idx)
            print(
                f"[eval] episode {episode_idx + 1}/{self.num_render_episodes} "
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
