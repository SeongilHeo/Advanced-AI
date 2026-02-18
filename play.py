import os 
import gymnasium as gym
import argparse
import pygame
import numpy as np

from environment.gym import get_env
from util.keyboard import get_action

TURNBASED = ["blackjack"]

def simple_run(env_name, key_map, make_kwargs, fps=30, mujoco=False):
    if make_kwargs.get("continuous"):
        make_kwargs["continuous"] = False
    env = gym.make(env_name, **make_kwargs, render_mode="human")
    obs, info = env.reset()

    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "100,100")

    pygame.init()
    # use pygame for keyboard input in mujoco envs
    if mujoco:
        screen = pygame.display.set_mode((400, 200))
        pygame.display.set_caption("MuJoCo Control")

    clock = pygame.time.Clock()

    running = True
    action = 0

    rewards = []
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        keys = pygame.key.get_pressed()

        action = get_action(keys, key_map)
        if action is None:
            # User requested quit (e.g., ESC). Clean up gracefully.
            running = False
            break
        
        if mujoco:
            action = np.array([action], dtype=np.float32)
            # Keep actions within the environment's valid range
            action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
        
        # Keep the pygame window responsive
        if mujoco:
            screen.fill((30, 30, 30))
            pygame.display.flip()
        
        rewards.append(rew)
        clock.tick(fps)

    env.close()
    pygame.quit()

    if rewards:
        print("Cumulative Reward:", float(np.sum(rewards)))
    return running  # False if user quit or episode ended

if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument("--env_name", type=str, default="CartPole", help="Select gym environment")
    params.add_argument("--fps", type=int, default=30, help="Select gym environment")

    args = params.parse_args()

    assert args.env_name.lower() not in TURNBASED, f"Sorry, not support turn-based game. {TURNBASED}"

    env_name, make_kwargs, key_map, mujoco = get_env(args.env_name, check_keymap=True, check_mujoco=True)

    assert key_map, f"Sorry, not support this game. {env_name}"

    running = True

    print("------Press ESCAPE to quit------")
    while running:
        running = simple_run(env_name, key_map, make_kwargs, fps=args.fps, mujoco=mujoco)
