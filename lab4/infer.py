import os
from multiprocessing import set_start_method
from typing import Tuple

import gym
import torch
import cv2
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

from model import PPO, PPORolloutDataset, worker_init


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision('high')
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    wandb.init(mode="disabled")

    env_name = "Skiing-v4"
    num_steps = 2048
    num_envs = 4
    max_epochs = 10
    batch_size = 64
    gamma = 0.99
    lam = 0.95

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    model = PPO.load_from_checkpoint('./lightning_logs/lemoowuo/checkpoints/best.ckpt')
    model.to(device)
    model.eval()

    env = gym.make(env_name)
    state = env.reset()

    done = False
    print('start')
    while not done:
        rendered_image = env.render(mode='rgb_array')
        cv2.imshow('Rendered Environment', rendered_image)
        cv2.waitKey(50)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            action_logits, _ = model(state_tensor)
            dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample().item()
        state, _, done, _ = env.step(action)

    env.close()
    cv2.destroyAllWindows()
    