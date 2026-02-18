from typing import Dict, List

from .common import Algorithm
from environment.gym import get_env
from model.a2c import MLPActorCritic

import numpy as np
import torch
from torch.optim import Adam

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

        self.lr=args.lr#1e-2, 
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

        self.optimizer = Adam(self.policy_net.parameters(), lr=self.lr)
        self.use_rtg=args.use_rtg

        self.train_returns: Dict[str, List[float]] = {"episode_rewards": [], "batch_length": [], "batch_points":[], "batch_rets_mean": [],  "batch_rets_std": []}
        self.eval_returns: Dict[str, List[float]] = {"points": [], "episode_rewards": []}
        
        # Global episode counter for deterministic per-episode seeding during training
        self.train_episode_counter = 0

    def _reward_to_go(self, rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs
    
    # make loss function whose gradient, for the right data, is policy gradient
    def _compute_loss(self, logp, weights):
        return -(logp * weights).mean()
    
    # for training policy
    def _train_one_epoch(self, env, idx):
        self.policy_net.train()
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables (deterministic per-episode seeds)
        if self.use_seed:
            episode_seed = self.seed + self.train_episode_counter
            obs, info = env.reset(seed=episode_seed)
        else:
            obs, info = env.reset()
        
        self.train_episode_counter += 1
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:
            # save obs
            batch_obs.append(obs.copy())
            # act in the environment
            obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32).to(self.device)
            # Sample an action (no grad needed for sampling)
            with torch.no_grad():
                act, _ = self.policy_net.act(obs_t)
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                
                self.train_returns["episode_rewards"].append(ep_ret)
                self.train_returns["batch_length"].append(ep_len)

                if self.use_rtg:
                    # the weight for each logprob(a_t|s_t) is reward-to-go from t
                    batch_weights += list(self._reward_to_go(ep_rews))
                else: 
                    # the weight for each logprob(a|s) is R(tau)
                    batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                if self.use_seed:
                    episode_seed = self.seed + self.train_episode_counter
                    obs, info = env.reset(seed=episode_seed)
                else:
                    obs, info = env.reset()
                self.train_episode_counter += 1
                done, ep_rews = False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > self.batch_size:
                    break

        # Compute log-probabilities with gradients from the current policy
        obs_batch = torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=self.device)
        act_batch = torch.as_tensor(np.array(batch_acts), dtype=torch.float32, device=self.device)
        weights_batch = torch.as_tensor(np.array(batch_weights), dtype=torch.float32, device=self.device)

        logp_batch = self.policy_net.log_prob(obs_batch, act_batch)

        # take a single policy gradient update step
        self.optimizer.zero_grad()
        batch_loss = self._compute_loss(logp_batch, weights_batch)
        batch_loss.backward()
        self.optimizer.step()
        
        return batch_loss, batch_rets, batch_lens
    
    def _train(self):
        env = self._make_env(render_mode=self.render_mode)
        if self.use_seed:
            env.action_space.seed(self.seed)

        # training loop
        for epoch_idx in range(self.num_epochs):
            batch_loss, batch_rets, batch_lens = self._train_one_epoch(env, epoch_idx)
            print(f"[train] epoch: {epoch_idx:3d} | loss: {batch_loss:>9.3f} | return: {np.mean(batch_rets):>9.3f} | ep_len: {np.mean(batch_lens):>9.3f}")

            self.train_returns["batch_rets_mean"].append(np.mean(batch_rets))
            self.train_returns["batch_rets_std"].append(np.std(batch_rets))
            self.train_returns["batch_points"].append(epoch_idx)


            if self.render_per_epoch:
                self._render()

        env.close()

        print("[Info] (pg) Save artifacts.")    
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
                obs = torch.as_tensor(np.array(obs), dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    act, logp = self.policy_net.act(obs)
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
