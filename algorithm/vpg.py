from typing import Dict, List

from .common import Algorithm
from environment.gym import get_env
from model.fcnet import FCNet

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from gymnasium.spaces import Discrete, Box

class VPG(Algorithm):
    def __init__(self, args):
        self.args=args
        self._init_artifact_paths(args)
        
        self.seed = getattr(args, "seed", None)
        self.use_seed = self.seed is not None
        if self.use_seed:
            self._seed_everything(self.seed)

        self.device = self._resolve_device(args.device)
        self.env_name, self.env_kwargs = get_env(args.env_name)
        self.render_mode = "human" if args.human_render else None
        self.mode = str(getattr(args, "mode", "train")).lower()
        self.render_per_epoch=args.render_per_epoch

        self.lr=args.lr#1e-2, 
        self.num_epochs=args.num_epochs#50
        self.batch_size=args.batch_size#5000
        self.hidden_dim=args.hidden_dim#32

        self.num_layers=args.num_layers
        self.activation=args.activation# "tahn",
        self.output_activation=args.output_activation#"identity"

        self.num_eval_episodes=args.num_eval_episodes

        warmup_env = self._make_env()
        assert isinstance(warmup_env.observation_space, Box), \
            "This example only works for envs with continuous state spaces."
        assert isinstance(warmup_env.action_space, Discrete), \
            "This example only works for envs with discrete action spaces."
        self.obs_dim=warmup_env.observation_space.shape[0]
        self.n_actions = warmup_env.action_space.n
        warmup_env.close()
        
        self.policy_net=FCNet(
            hidden_dim=self.hidden_dim,
            input_dim=self.obs_dim,
            output_dim=self.n_actions,
            num_layers=self.num_layers,
            activation=self.activation,
            output_activation=self.output_activation
        )
        self.policy_net.to(self.device)

        self.optimizer = Adam(self.policy_net.parameters(), lr=self.lr)
        
        self.train_returns: Dict[str, List[float]] = {"episode_rewards": [], "batch_length": [], "batch_points":[], "batch_rets_mean": [],  "batch_rets_std": []}
        self.eval_returns: Dict[str, List[float]] = {"points": [], "episode_rewards": []}
        
        # Global episode counter for deterministic per-episode seeding during training
        self._train_episode_counter = 0

        self.use_rtg=args.use_rtg

    def reward_to_go(self, rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    # make function to compute action distribution
    def _get_policy(self, obs):
        logits = self.policy_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def _get_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)
        return self._get_policy(obs).sample().item()
    
    # make loss function whose gradient, for the right data, is policy gradient
    def _compute_loss(self, obs, act, weights):
        logp = self._get_policy(obs).log_prob(act)
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
            episode_seed = self.seed + self._train_episode_counter
            obs, info = env.reset(seed=episode_seed)
        else:
            obs, info = env.reset()
        
        self._train_episode_counter += 1
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:
            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = self._get_action(obs)
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
                    batch_weights += list(self.reward_to_go(ep_rews))
                else: 
                    # the weight for each logprob(a|s) is R(tau)
                    batch_weights += [ep_ret] * ep_len


                # reset episode-specific variables
                if self.use_seed:
                    episode_seed = self.seed + self._train_episode_counter
                    obs, info = env.reset(seed=episode_seed)
                else:
                    obs, info = env.reset()
                self._train_episode_counter += 1
                done, ep_rews = False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > self.batch_size:
                    break
                

        # take a single policy gradient update step
        self.optimizer.zero_grad()
        batch_loss = self._compute_loss(
            obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=self.device),
            act=torch.as_tensor(np.array(batch_acts), dtype=torch.long, device=self.device),
            weights=torch.as_tensor(np.array(batch_weights), dtype=torch.float32, device=self.device),
        )
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
            print(f"[train] epoch: {epoch_idx:3d} | loss: {batch_loss:>7.3f} | return: {np.mean(batch_rets):>7.3f} | ep_len: {np.mean(batch_lens):>7.3f}")

            self.train_returns["batch_rets_mean"].append(np.mean(batch_rets))
            self.train_returns["batch_rets_std"].append(np.std(batch_rets))
            self.train_returns["batch_points"].append(epoch_idx)


            if self.render_per_epoch:
                self._render()

        env.close()
        print("[Info] (pg) Save artifacts.")    
        self._save_artifacts()
            
    def _render(self):
        self._evaluate(render_mode="human", store=False)

    def _evaluate(self, render_mode=None, store: bool = True):
        env = self._make_env(render_mode=render_mode)
        if self.use_seed:
            env.action_space.seed(self.seed)
        self.policy_net.eval()

        for episode_idx in range(self.num_eval_episodes):
            if self.use_seed:
                obs, info = env.reset(seed=self.seed + episode_idx)
            else:
                obs, info = env.reset()
            
            done = False
            total_reward = 0.0

            while not done:
                with torch.no_grad():
                    act = self._get_action(obs)
                obs, rew, terminated, truncated, info = env.step(act)
                done = terminated or truncated
                total_reward += rew

            print(f"[eval] episode {episode_idx+1}/{self.num_eval_episodes} total reward: {total_reward}")
            self.eval_returns["episode_rewards"].append(total_reward)

        env.close()

    def _build_model_state(self) -> Dict:
        return {
            "n_actions": self.n_actions,
            "policy_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def _load_model_state(self, model_state: Dict):
        loaded_n_actions = int(model_state.get("n_actions", self.n_actions))
        if loaded_n_actions != self.n_actions:
            raise ValueError("Loaded model action-space size does not match environment.")
        if "policy_state_dict" not in model_state:
            raise ValueError("Checkpoint missing policy_state_dict.")

        self.policy_net.load_state_dict(model_state["policy_state_dict"])
        if model_state.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(model_state["optimizer_state_dict"])
        self.policy_net.eval()

    def _load_checkpoint(self):
        payload = self._load_model_checkpoint()
        model_state = payload.get("model_state", payload)
        self._load_model_state(model_state)
        return payload

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
