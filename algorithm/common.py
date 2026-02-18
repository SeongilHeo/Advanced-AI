from abc import ABC
from datetime import datetime
from pathlib import Path
import pickle
import json
from typing import Any, Dict, Optional

import gymnasium as gym
import torch
import random
import numpy as np

class Algorithm(ABC):
    def _resolve_device(self, device_name: str) -> torch.device:
        requested = str(device_name).lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[Warn] (dqn) CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        if requested.startswith("mps") and not torch.backends.mps.is_available():
            print("[Warn] (dqn) MPS requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)
    
    def _seed_everything(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _make_env(self, render_mode=None, **make_kwargs):
        if not hasattr(self, "env_name"):
            raise AttributeError(
                "Algorithm._make_env requires self.env_name to be set before use."
            )

        env_kwargs = dict(getattr(self, "env_kwargs", {}))
        env_kwargs.update(make_kwargs)
        if render_mode is not None:
            env_kwargs["render_mode"] = render_mode

        return gym.make(self.env_name, **env_kwargs)

    def _init_artifact_paths(self, args):
        self.output_dir = Path(getattr(args, "output_dir", "output"))
        self.hparams_path = Path(getattr(args, "hparams_path", "hparams.json"))
        self.checkpoint_path = Path(getattr(args, "checkpoint_path", "checkpoint.pkl"))
        self.results_path = Path(getattr(args, "results_path", "results.pkl"))
        self.run_id = getattr(args, "run_id", None)
        self.run_dir: Optional[Path] = None

    def _configure_output_paths(self, create_run_dir: bool = False) -> Optional[Path]:
        if create_run_dir:
            run_id = self.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.output_dir / run_id
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.hparams_path = self.run_dir / self.hparams_path.name
            self.checkpoint_path = self.run_dir / self.checkpoint_path.name
            self.results_path = self.run_dir / self.results_path.name
            return self.run_dir

        self.hparams_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        return None
    
    def _save_json_payload(self, path: Path, payload: Dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        return path
    
    def _save_pickle_payload(self, path: Path, payload: Dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(payload, file)
        return path
    
    def _load_json_payload(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_pickle_payload(self, path: Path) -> Dict[str, Any]:
        with path.open("rb") as file:
            return pickle.load(file)
        
    def _save_hparams(
        self,
        hparams: Dict[str, Any],
        hparams_path: Optional[Path] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = self.hparams_path if hparams_path is None else Path(hparams_path)
        
        exc_hparams, inc_hparams = hparams

        payload: Dict[str, Any] = {
            "format_version": 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "exc_hparams": exc_hparams,
            "inc_hparams": inc_hparams,
        }
        if extra_payload:
            payload.update(extra_payload)
        return self._save_json_payload(path, payload)
    
    def _save_model_checkpoint(
        self,
        model_state: Dict[str, Any],
        checkpoint_path: Optional[Path] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = self.checkpoint_path if checkpoint_path is None else Path(checkpoint_path)
        payload: Dict[str, Any] = {
            "format_version": 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_state": model_state,
        }
        if extra_payload:
            payload.update(extra_payload)
        return self._save_pickle_payload(path, payload)

    def _load_model_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        path = self.checkpoint_path if checkpoint_path is None else Path(checkpoint_path)
        return self._load_pickle_payload(path)

    def _save_training_results(
        self,
        results: Dict[str, Any],
        results_path: Optional[Path] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = self.results_path if results_path is None else Path(results_path)
        payload: Dict[str, Any] = {
            "format_version": 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "results": results,
        }
        if extra_payload:
            payload.update(extra_payload)
        return self._save_pickle_payload(path, payload)
    
    def _build_hparams(self):
        _INCLUSIVE_HPARAM_KEYS = ("seed", "use_seed", "num_eval_episodes", "device", "mode")
        inc_hparams = {}
        exc_hparams = dict(vars(self.args))
        for key in _INCLUSIVE_HPARAM_KEYS:
            inc_hparams[key] = exc_hparams.pop(key)

        return exc_hparams, inc_hparams

    def _build_model_state(self) -> Dict:
        return {
            "policy_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def _load_model_state(self, model_state: Dict):
        self.policy_net.load_state_dict(model_state["policy_state_dict"])
        if model_state.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(model_state["optimizer_state_dict"])

    def _load_checkpoint(self):
        payload = self._load_model_checkpoint()
        model_state = payload.get("model_state", payload)
        self._load_model_state(model_state)
        return payload
    
    def _save_artifacts(self):
        run_dir = self._configure_output_paths(create_run_dir=True)
        print(f"[Info] Run directory: {run_dir}")

        self._save_hparams(
            hparams=self._build_hparams()
        )
        self._save_model_checkpoint(
            model_state=self._build_model_state()
        )
        self._save_training_results(
            results={
                "train_return": self.train_returns,
                "eval_return": self.eval_returns,
            }
        )
        print(f"[Info] Saved hparams to {self.hparams_path}")
        print(f"[Info] Saved checkpoint to {self.checkpoint_path}")
        print(f"[Info] Saved results to {self.results_path}")
