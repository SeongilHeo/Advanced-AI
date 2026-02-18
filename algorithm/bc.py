from .common import Algorithm

import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from model.fcnet import FCNet

from util.teleop import collect_demos
from environment.gym import get_env

class BC(Algorithm):
    def __init__(self, args):
        self.args = args
        # envirnment parameters
        self.device=args.device
        self.env_name, self.env_kwargs, self.key_map = get_env(args.env_name, check_keymap=True)
        self.key_map.pop("noop", None)
        self.render_mode= "human" if args.human_render else None
        self.mode = str(getattr(args, "mode", "train")).lower()
        self.render_episodes = int(getattr(args, "render_episodes", 5))
        self._init_artifact_paths(args)

        # algorithm parameters
        self.num_demos=args.num_demos
        self.num_bc_iters=args.num_bc_iters
        self.num_evals=args.num_evals

        # model parameters
        self.hidden_dim=args.hidden_dim # 64
        self.input_dim=args.input_dim # 2
        self.output_dim=args.output_dim # 3
        self.num_layers=args.num_layers # 0
        self.activation=args.activation # "relu"
        # model
        self.policy = FCNet(
            hidden_dim=self.hidden_dim,
            input_dim=self.input_dim,
            output_dim=self.output_dim, 
            num_layers=self.num_layers, 
            activation=self.activation
        )
        self.policy.to(self.device)
        
        # train parameters
        self.lr = args.lr
        self.weight_decay=args.weight_decay
        # self.optimizer=args.optimizer # Adam

        self.optimizer = Adam(self.policy.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # return storage
        self.train_returns = {}
        self.eval_returns = {}
    
    def _collect_human_demos(self,):
        # collect demo form env
        env = self._make_env(render_mode='rgb_array') 
        demos = collect_demos(env, keys_to_action=self.key_map, num_demos=self.num_demos, noop=1)
        env.close()

        # torchify
        states, actions, next_states = zip(*demos)

        states = np.stack(states)         # shape: (N, 2)
        actions = np.array(actions)       # shape: (N,)
        next_states = np.stack(next_states)  # shape: (N, 2)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        return states, actions, next_states
    
    def _train_policy(self, obs, acs):
        self.policy.train()
        
        self.train_returns["loss"]=[]

        for _ in range(self.num_bc_iters):
            logits = self.policy(obs)
            loss = F.cross_entropy(logits, acs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_returns["loss"].append(float(loss.item()))
            
        self.policy.eval()

    def _evaluate_policy(self, num_episodes=None, render_mode=None):
        episodes = self.num_evals if num_episodes is None else int(num_episodes)
        env = self._make_env(render_mode=render_mode)

        self.eval_returns["episode_reward"] = []
        self.policy.eval()
        with torch.no_grad():
            for episode_idx in range(episodes):
                done = False
                total_reward = 0.0
                obs, _ = env.reset()
                while not done:
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    action = torch.argmax(self.policy(obs_tensor), dim=1).item()
                    obs, rew, terminate, turncated, _ = env.step(action)
                    done = terminate or turncated
                    total_reward += rew
                self.eval_returns["episode_reward"].append(total_reward)
                print(f"[eval] episode {episode_idx+1}/{episodes} total reward: {total_reward}")
        
        env.close()
        if self.eval_returns["episode_reward"]:
            print("[Summary] "+
                f"min: {np.min(self.eval_returns["episode_reward"])} \t"+
                f"max: {np.max(self.eval_returns["episode_reward"])} \t" +
                f"mean: {np.mean(self.eval_returns["episode_reward"])}",
            )

    def _build_model_state(self):
        return {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "env_name": self.env_name,
        }

    def _load_artifacts(self):
        payload = self._load_model_checkpoint()
        model_state = payload.get("model_state", payload)

        if "policy_state_dict" not in model_state:
            raise ValueError("Checkpoint missing policy_state_dict.")

        self.policy.load_state_dict(model_state["policy_state_dict"])
        if model_state.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(model_state["optimizer_state_dict"])
        self.policy.eval()
        return payload

    def _train(self):
        print("[Info] (bc) Collect human demonstrations.")
        obs, acs, _ = self._collect_human_demos()

        print("[Info] (bc) Train policy.")
        self._train_policy(obs, acs)

        print("[Info] (bc) Evaluate policy.")
        self._evaluate_policy(num_episodes=self.num_evals, render_mode=self.render_mode)

        print("[Info] (bc) Save artifacts.")
        self._save_artifacts()

    def _evaluate(self):
        print("[Info] (bc) Load checkpoint.")
        payload = self._load_artifacts()

        print("[Info] (bc) Evaluate policy.")
        self._evaluate_policy(num_episodes=self.num_evals, render_mode=None)

        print(f"[Info] (bc) Loaded checkpoint from {self.checkpoint_path}")
        if "hparams" in payload:
            print(f"[Info] (bc) Loaded hparams keys: {sorted(payload['hparams'].keys())}")

    def _render(self):
        print("[Info] (bc) Load checkpoint.")
        payload = self._load_artifacts()

        print("[Info] (bc) Render policy.")
        self._evaluate_policy(num_episodes=self.render_episodes, render_mode="human")

        print(f"[Info] (bc) Loaded checkpoint from {self.checkpoint_path}")
        if "hparams" in payload:
            print(f"[Info] (bc) Loaded hparams keys: {sorted(payload['hparams'].keys())}")

    def process(self,):
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
