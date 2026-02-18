import pygame

env_table = {
    "mountaincar" : {
        "env": "MountainCar-v0",
        "make_kwargs": {},
        "action": {
            (pygame.K_LEFT): 0, 
            (pygame.K_RIGHT): 2,
            ("noop"): 1

        }
    },
    "blackjack": {
        "env": "Blackjack-v1",
        "make_kwargs": {},
        "action": {
            (pygame.K_h): 0, 
            (pygame.K_s): 1
        }
    },
    "lunarlander" : {
        "env": "LunarLander-v3",
        "make_kwargs": {
            "continuous": False,
            "gravity": -10.0,
            "enable_wind": False,
            "wind_power": 15.0,
            "turbulence_power": 1.5,
        },
        "action": {
            (pygame.K_LEFT): 1, 
            (pygame.K_UP): 2, 
            (pygame.K_RIGHT): 3
        }
    },
    "carracing" : {
        "env": "CarRacing-v3",
        "make_kwargs": {
            "continuous": True,
        },
        "action": {
            (pygame.K_RIGHT): 1,
            (pygame.K_LEFT): 2, 
            (pygame.K_UP): 3, 
            (pygame.K_DOWN): 4
        }
    },
    "cartpole" : {
        "env": "CartPole-v1",
        "make_kwargs": {},
        "action": {
            (pygame.K_LEFT): 0, 
            (pygame.K_RIGHT): 1
        }
    },
    "invertedpendulum" : {
        "env": "InvertedPendulum-v5",
        "make_kwargs": {
            "reset_noise_scale": 0.1,
            "max_episode_steps": 1000,
        },
        "action": {
            (pygame.K_LEFT): -1.0, 
            (pygame.K_RIGHT): 1.0
        },
        "mujoco": True,
    },
    "hopper" : {
        "env": "Hopper-v5",
        "make_kwargs": {
            "ctrl_cost_weight": 1e-3,
        },
        "mujoco": True,
    },
    "walker2d" : {
        "env": "Walker2d-v5",
        "make_kwargs": {
            "ctrl_cost_weight": 1e-3,
        },
        "mujoco": True,
    },
    "halfcheetah" : {
        "env": "HalfCheetah-v5",
        "make_kwargs": {
            "ctrl_cost_weight": 0.1,
        },
        "mujoco": True,
    },

}

def get_env(name, check_keymap=False, check_mujoco=False):
    env_spec = env_table[name.lower()]
    
    env_name = env_spec["env"]
    make_kwargs = env_spec.get("make_kwargs", {})
    keymap = env_spec.get("action", {})
    mujoco = env_spec.get("mujoco", False)

    if check_keymap and check_mujoco:
        return env_name, make_kwargs, keymap, mujoco
    elif check_keymap:
        return env_name, make_kwargs, keymap
    elif check_mujoco:
        return env_name, make_kwargs, mujoco
    
    return env_name, make_kwargs
