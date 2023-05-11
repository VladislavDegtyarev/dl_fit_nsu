import os
from typing import Tuple


import gym
import torch
import numpy as np
from torch.utils.data import IterableDataset
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple
import torch.optim as optim
import wandb


class PPORolloutDataset(IterableDataset):
    def __init__(self, env, model, num_steps, device, batch_size=64, gamma=0.99, lam=0.95):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps = num_steps
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam

    def __iter__(self):
        return ppo_data_loader(self.env, self.model, self.num_steps, self.device, self.batch_size, self.gamma, self.lam)


def worker_init(worker_id):
    gym.make("Skiing-v4").seed(torch.randint(0, 1000, ()).item())  
    

class Policy(nn.Module):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_output_dim = self._get_conv_output(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(conv_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def _get_conv_output(self, input_shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *input_shape)).shape[1]

    def forward(self, state):
        features = self.conv(state)
        action_logits = self.policy(features)
        return action_logits
    
    
class Value(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_output_dim = self._get_conv_output(input_shape)

        self.value = nn.Sequential(
            nn.Linear(conv_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _get_conv_output(self, input_shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *input_shape)).shape[1]

    def forward(self, state):
        features = self.conv(state)
        value = self.value(features)
        return value


class PPO(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        action_dim,
        gamma=0.5,
        lr=1e-3,
        clip_epsilon=0.1,
        update_epochs=1,
        entropy_coeff=0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.best_loss = float("inf")
        self.best_model_path = os.path.join(os.getcwd(), "best_model.pt")

        self.policy = Policy(input_shape, action_dim)
        self.value = Value(input_shape)    
    

    def training_step(self, batch, batch_idx):
        states, actions, rewards, dones, values, log_probs_old, returns, advantages = batch
        state_tensor = states.permute(0, 3, 1, 2).to(self.device)
        action_tensor = actions.to(self.device)
        returns_tensor = returns.unsqueeze(-1).to(self.device)
        advantages_tensor = advantages.unsqueeze(-1).to(self.device)
        log_probs_old_tensor = log_probs_old.unsqueeze(-1).to(self.device)

        log_probs_old = log_probs_old_tensor.detach()
        advantages = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        for _ in range(self.hparams.update_epochs):
            action_logits = self.policy(state_tensor)
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(action_tensor)
            entropy = dist.entropy().mean()
            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.hparams.clip_epsilon, 1 + self.hparams.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            values = self.value(state_tensor)
            critic_loss = F.mse_loss(returns_tensor, values)

            loss = actor_loss + critic_loss - self.hparams.entropy_coeff * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer_step(self.current_epoch, batch_idx, self.optimizer)

        self.log("actor_loss", actor_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("critic_loss", critic_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        episode_reward = sum(rewards).item()
        self.log("episode_reward", episode_reward, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def forward(self, state):
        action_logits = self.policy(state)
        value = self.value(state)
        return action_logits, value

    def calculate_returns(self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        not_dones = 1 - dones
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = values[-1]
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.hparams.gamma * next_value * not_dones[t] - values[t]
            gae = delta + self.hparams.gamma * self.hparams.gae_lambda * not_dones[t] * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        return returns, advantages
    
    
    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return self.optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx=None):
        optimizer.zero_grad(set_to_none=True)

        
def normalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)

    
def ppo_data_loader(env, model, num_steps, device, batch_size=64, gamma=0.99, lam=0.95):
    state = env.reset()

    while True:
        states, actions, rewards_batch, dones, values, log_probs_old = [], [], [], [], [], []
        for _ in range(num_steps):
            states.append(state)
            state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                action_logits, value = model(state_tensor)
                dist = Categorical(logits=action_logits)
                action = dist.sample().squeeze().item()
                log_probs_old.append(dist.log_prob(torch.tensor(action, dtype=torch.long, device=device)).item())
            state, reward, done, _ = env.step(action)
            actions.append(action)
            rewards_batch.append(reward)
            dones.append(done)
            values.append(value.item())
            if done:
                state = env.reset()

        rewards_normalized = normalize(torch.tensor(rewards_batch, dtype=torch.float32))
        rewards = rewards_normalized

        # Calculate returns and advantages
        last_value = values[-1] if not done else 0
        returns = []
        advantages = []
        R = last_value
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            R = r + gamma * (1 - d) * R
            returns.append(R)
            advantages.append(R - v)
        returns = list(reversed(returns))
        advantages = list(reversed(advantages))

        # Create mini-batches
        for idx in range(0, num_steps, batch_size):
            batch_end = idx + batch_size
            yield (
                torch.tensor(np.array(states[idx:batch_end]), dtype=torch.float32),
                torch.tensor(np.array(actions[idx:batch_end]), dtype=torch.long),
                torch.tensor(np.array(rewards[idx:batch_end]), dtype=torch.float32),
                torch.tensor(np.array(dones[idx:batch_end]), dtype=torch.float32),
                torch.tensor(np.array(values[idx:batch_end]), dtype=torch.float32),
                torch.tensor(np.array(log_probs_old[idx:batch_end]), dtype=torch.float32),
                torch.tensor(np.array(returns[idx:batch_end]), dtype=torch.float32),
                torch.tensor(np.array(advantages[idx:batch_end]), dtype=torch.float32),
            )