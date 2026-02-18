from typing import Dict, List

from .common import Algorithm
from environment.gym import get_env
from model.ac import MLPActorCritic

import numpy as np
import torch
from torch.optim import Adam
from util.buffer import Buffer

class VPG(Algorithm):
    def __init__(self, args):
        self.args=args
        self._init_artifact_paths(args)
        
        self.seed = args.seed
        self.use_seed = args.use_seed
        if self.use_seed:
            self._seed_everything(self.seed)

        self.device = self._resolve_device(args.device)
        self.env_name, self.env_kwargs= get_env(args.env_name)
        self.mode = args.mode

        self.render_mode = "human" if args.human_render else None
        self.render_per_epoch=args.render_per_epoch

        self.pi_lr=args.pi_lr#1e-2, 
        self.vf_lr=args.vf_lr#1e-2, 
        self.num_epochs=args.num_epochs#50
        self.batch_size=args.batch_size#5000
        self.num_eval_episodes=args.num_eval_episodes

        self.hidden_dims=args.hidden_dims#32
        self.activation=args.activation#"tahn",
        self.output_activation=args.output_activation#"identity"

        warmup_env = self._make_env()
        self.observation_space = warmup_env.observation_space
        self.action_space = warmup_env.action_space
        warmup_env.close()
        
        self.policy_net=MLPActorCritic(
            self.observation_space,
            self.action_space,
            self.hidden_dims,
            activation=self.activation,
        )
        self.policy_net.to(self.device)

        # Optimizers (policy and value)
        self.pi_optimizer = Adam(self.policy_net.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.policy_net.v.parameters(), lr=self.vf_lr)
        self.train_v_iters = args.train_v_iters

        # GAE params
        self.gamma = args.gamma
        self.lam = args.lam

        # Buffer dims
        obs_dim = self.observation_space.shape
        act_dim = self.action_space.shape

        self.buffer = Buffer(
            obs_dim,
            act_dim,
            self.batch_size,
            gamma=self.gamma,
            lam=self.lam,
            device=self.device,
        )

        self.train_returns: Dict[str, List[float]] = {
            "epoch": [],
            "episode_rewards": [], 
            "episode_lengths": [], 
            "loss_pi": [],
            "loss_v": [],
            "kl": [],
            "ent": [],
            "d_loss_pi": [],
            "d_loss_v": [],
            "epoch_episode_rewards": [],
            "epoch_episode_lengths": [],
            "epoch_episode_counter": [],
            "epoch_v_vals": [],
        }
        self.eval_returns: Dict[str, List[float]] = {
            "points": [], 
            "episode_rewards": []
        }
        
        # Global episode counter for deterministic per-episode seeding during training
        self.train_episode_counter = 0

    def _build_model_state(self) -> Dict:
        return {
            "policy_state_dict": self.policy_net.state_dict(),
            "pi_optimizer_state_dict": self.pi_optimizer.state_dict(),
            "vf_optimizer_state_dict": self.vf_optimizer.state_dict(),
        }

    def _load_model_state(self, model_state: Dict):
        if "policy_state_dict" not in model_state:
            raise ValueError("Checkpoint missing policy_state_dict.")

        self.policy_net.load_state_dict(model_state["policy_state_dict"])
        if model_state.get("pi_optimizer_state_dict") is not None:
            self.pi_optimizer.load_state_dict(model_state["pi_optimizer_state_dict"])
        if model_state.get("vf_optimizer_state_dict") is not None:
            self.vf_optimizer.load_state_dict(model_state["vf_optimizer_state_dict"])

    def _reward_to_go(self, rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs
    
    # Set up function for computing VPG policy loss
    def _compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.policy_net.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def _compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.policy_net.v(obs) - ret)**2).mean()

    def _update(self):
        data = self.buffer.get()

        pi_loss_old, pi_info_old = self._compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = self._compute_loss_v(data).item()

        self.pi_optimizer.zero_grad()
        pi_loss, pi_info = self._compute_loss_pi(data)
        pi_loss.backward()
        self.pi_optimizer.step()

        # Value function learning
        for _ in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            v_loss = self._compute_loss_v(data)
            v_loss.backward()
            self.vf_optimizer.step()

        with torch.no_grad():
            pi_loss_new, pi_info_new = self._compute_loss_pi(data)
            v_loss_new = self._compute_loss_v(data)

        kl, ent = pi_info_new['kl'], pi_info_new['ent']
        self.train_returns['loss_pi'].append(pi_loss_old)
        self.train_returns['loss_v'].append(v_loss_old)
        self.train_returns['kl'].append(kl)
        self.train_returns['ent'].append(ent)
        self.train_returns['d_loss_pi'].append(pi_loss_new.item() - pi_loss_old)
        self.train_returns['d_loss_v'].append(v_loss_new.item() -v_loss_old)

    # for training policy
    def _train_one_epoch(self, env):
        self.policy_net.train()
        episode_rewards = []
        episode_lengths = []
        epoch_v_vals = []

        # reset episode-specific variables (deterministic per-episode seeds)
        if self.use_seed:
            episode_seed = self.seed + self.train_episode_counter
            obs, info = env.reset(seed=episode_seed)
        else:
            obs, info = env.reset()

        self.train_episode_counter += 1
        ep_ret = 0.0
        ep_len = 0

        # Collect experience until buffer is full
        while not self.buffer.is_full():

            with torch.no_grad():
                act, val, logp = self.policy_net.step(
                    torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                )

            next_obs, rew, terminated, truncated, info = env.step(act)
            ep_ret += rew
            ep_len += 1

            self.buffer.store(obs, act, rew, val, logp)
            epoch_v_vals.append(val)

            obs = next_obs

            epoch_ended = self.buffer.is_full()

            if terminated or truncated or epoch_ended:
                # Bootstrap if trajectory didn't reach a terminal state
                if truncated or (epoch_ended and not terminated):
                    with torch.no_grad():
                        _, last_val, _ = self.policy_net.step(
                            torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                        )
                else:
                    last_val = 0.0

                self.buffer.finish_path(last_val)

                # Only log completed episodes (terminal OR timeout)
                if terminated or truncated:
                    episode_rewards.append(ep_ret)
                    episode_lengths.append(ep_len)

                    # Reset env for next episode
                    if self.use_seed:
                        episode_seed = self.seed + self.train_episode_counter
                        obs, info = env.reset(seed=episode_seed)
                    else:
                        obs, info = env.reset()
                    self.train_episode_counter += 1

                    ep_ret = 0.0
                    ep_len = 0

        # Update policy/value using the full buffer
        self._update()
        
        self.train_returns["episode_rewards"].extend(episode_rewards)
        self.train_returns["episode_lengths"].extend(episode_lengths)
        self.train_returns['epoch_episode_rewards'].append(np.mean(episode_rewards))
        self.train_returns['epoch_episode_lengths'].append(np.mean(episode_lengths))
        self.train_returns['epoch_episode_counter'].append(self.train_episode_counter)
        self.train_returns['epoch_v_vals'].append(np.mean(epoch_v_vals))
    
    def _train(self):
        env = self._make_env(render_mode=self.render_mode)
        if self.use_seed:
            env.action_space.seed(self.seed)

        print("\n","-"*255)
        for k in self.train_returns.keys():
            if not k.startswith("episode_"):
                print(f"{k:^21}", end="\t")
        print("\n","-"*255)

        # training loop
        for epoch_idx in range(self.num_epochs):
            self._train_one_epoch(env)
            
            self.train_returns['epoch'].append(epoch_idx)

            if self.render_per_epoch:
                self._render()
            
            for k, v in self.train_returns.items():
                if not k.startswith("episode_"):
                    print(f"{v[-1]:^21.5f}", end="\t")
            print()

        print("\n","-"*255)

        env.close()

        print("[Info] (vpg) Save artifacts.")    
        self._save_artifacts()
            
    def _render(self):
        self._evaluate(render_mode="human")

    def _evaluate(self, render_mode=None):
        env = self._make_env(render_mode=render_mode)
        if self.use_seed:
            env.action_space.seed(self.seed)
        
        self._load_checkpoint()
        self.policy_net.eval()

        for episode_idx in range(self.num_eval_episodes):
            if self.use_seed:
                obs, info = env.reset(seed=self.seed + episode_idx)
            else:
                obs, info = env.reset()
            
            done = False
            total_reward = 0.0

            while not done:
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    act = self.policy_net.act(obs)
                obs, rew, terminated, truncated, info = env.step(act)
                done = terminated or truncated
                total_reward += rew

            print(f"[eval] episode {episode_idx+1}/{self.num_eval_episodes} total reward: {total_reward}")
            self.eval_returns["episode_rewards"].append(total_reward)

        env.close()

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
