import os
from multiprocessing import set_start_method
from typing import Tuple

import gym
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from model import PPO, PPORolloutDataset, worker_init


if __name__ == "_main_":
    set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision('medium')
    wandb.login()

    checkpoint_callback = ModelCheckpoint(
        filename='best.ckpt',
        save_top_k=1,
        verbose=True,
        monitor='episode_reward',
        mode='max',
        every_n_train_steps=100,
    )

    wandb.init(project="ppo_atari_skiing")
    env_name = "Skiing-v4"
    num_steps = 512
    num_envs = 4
    max_epochs = 10
    batch_size = 10
    gamma = 0.99
    lam = 0.95

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    model = PPO((3, 210, 160), env.action_space.n).to(device)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epochs,
                         logger=pl.loggers.WandbLogger(), log_every_n_steps=10,
                         callbacks=[checkpoint_callback])

    dataset = PPORolloutDataset(env, model, num_steps=num_steps, device=device, batch_size=batch_size, gamma=gamma, lam=lam)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_envs, worker_init_fn=worker_init, pin_memory=True)
    trainer.fit(model, dataloader)
