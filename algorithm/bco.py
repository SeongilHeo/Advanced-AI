from .bc import BC

import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from model.fcnet import FCNet

class BCO(BC):
    def __init__(self, args):
        super().__init__(args)

        # algorithm parameters
        self.num_random_iters=args.num_random_iters # 10
        self.num_inv_train_iters=args.num_inv_train_iters # 2000
        
        # inverse dynamic model 
        # model parameters
        self.inv_hidden_dim=args.inv_hidden_dim # 8
        self.inv_input_dim=args.inv_input_dim # 4
        self.inv_output_dim=args.inv_output_dim # 3
        self.inv_num_layers=args.inv_num_layers # 1
        self.inv_activation=args.inv_activation #"leaky_relu",
        
        self.inv_dyn = FCNet(
            hidden_dim=self.inv_hidden_dim,
            input_dim=self.inv_input_dim,
            output_dim=self.inv_output_dim,
            num_layers=self.inv_num_layers,
            activation=self.inv_activation 
        )
        self.inv_dyn.to(self.device)

        # train paramerters
        self.inv_lr=args.inv_lr # 1e-2
        self.inv_weight_decay=args.inv_weight_decay

        self.inv_optimizer = Adam(self.inv_dyn.parameters(), lr=self.inv_lr, weight_decay=self.inv_weight_decay)

        # return storage
        self.train_returns = {}
        self.eval_returns = {}

    def _build_model_state(self):
        model_state = super()._build_model_state()
        model_state.update(
            {
                "inv_dyn_state_dict": self.inv_dyn.state_dict(),
                "inv_optimizer_state_dict": self.inv_optimizer.state_dict(),
            }
        )
        return model_state

    def _load_artifacts(self):
        payload = super()._load_artifacts()
        model_state = payload.get("model_state", payload)

        if "inv_dyn_state_dict" not in model_state:
            raise ValueError("Checkpoint missing inv_dyn_state_dict for BCO.")

        self.inv_dyn.load_state_dict(model_state["inv_dyn_state_dict"])
        if model_state.get("inv_optimizer_state_dict") is not None:
            self.inv_optimizer.load_state_dict(model_state["inv_optimizer_state_dict"])
        self.inv_dyn.eval()
        return payload

    def _collect_random_interaction_data(self,):
        state_next_state = []
        actions = []
        env = self._make_env()
        for _ in range(self.num_random_iters):
            obs, _ = env.reset()

            terminate, turncated = False, False
            while not (terminate or turncated):
                a = env.action_space.sample()
                next_obs, reward, terminate, turncated, info = env.step(a)
                state_next_state.append(np.concatenate((obs,next_obs), axis=0))
                actions.append(a)
                obs = next_obs

        env.close()

        s_s2, acs = np.array(state_next_state), np.array(actions)
        s_s2_torch = torch.from_numpy(np.array(s_s2)).float().to(self.device)
        a_torch = torch.from_numpy(np.array(acs)).long().to(self.device)

        return s_s2_torch, a_torch
    
    def _train_inv_dyn(self,s_s2_torch,a_torch):
        self.inv_dyn.train()
        
        #train inverse dynamics model
        for _ in range(self.num_inv_train_iters):
            self.inv_optimizer.zero_grad()
            logits = self.inv_dyn(s_s2_torch)
            loss = F.cross_entropy(logits, a_torch)
            loss.backward()
            self.inv_optimizer.step()
            
        self.inv_dyn.eval()

    def _train(self):
        print("[Info] (bco) Collect random interaction data.")
        s_s2_torch, a_torch = self._collect_random_interaction_data()
        
        print("[Info] (bco) Train inverse dynamic model.")
        self._train_inv_dyn(s_s2_torch, a_torch)

        print("[Info] (bco) Collect human demonstrations")
        obs, acs, obs2 = self._collect_human_demos()

        print("[Info] (bco) Predict actions of human demonstration via inverse dynamic model.")
        state_trans = torch.cat((obs, obs2), dim = 1)
        outputs = self.inv_dyn(state_trans)
        _, acs = torch.max(outputs, 1)
        
        print("[Info] (bco) Train policy.")
        self._train_policy(obs, acs)
        
        print("[Info] (bco) Evaluate policy")
        self._evaluate_policy(num_episodes=self.num_evals, render_mode=self.render_mode)
        
        print("[Info] (bco) Save artifacts.")
        self._save_artifacts()

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