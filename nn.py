import numpy as np
from dataclasses import dataclass
from config import *
from pingpong import State

import torch
from typing import List
from torch import nn
from copy import deepcopy


def add_noise(action, noise_scale):
    noise = np.random.normal(0, noise_scale)
    return np.clip(action + noise, -PADDLE_VELOCITY_MAX, PADDLE_VELOCITY_MAX)


@dataclass 
class Observation:
    ball_vx: float          
    ball_vy: float          

    paddle_vy: float
    paddle_y: float

    ball_x_distance_to_paddle: float
    ball_y_distance_to_paddle: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.ball_vx / MAX_VELOCITY,           # 0
            self.ball_vy / MAX_VELOCITY,           # 1
            self.paddle_vy / MAX_VELOCITY,         # 2
            self.paddle_y / MAX_Y,                 # 3
            self.ball_x_distance_to_paddle / MAX_X, # 4
            self.ball_y_distance_to_paddle / MAX_Y, # 5
        ])


@dataclass
class ObservationBuffer:
    observations: List[Observation]

    def initialize(self):
        for i in range(NN_OBSERVATIONS):
            self.observations.append(Observation(
                ball_vx=0.0, 
                ball_vy=0.0, 
                paddle_vy=0.0, 
                paddle_y=MAX_Y/2,
                ball_x_distance_to_paddle=0.0,  
                ball_y_distance_to_paddle=0.0, 
            ))

    def add(self, observation):
        self.observations.pop(0)
        self.observations.append(observation)
    
    def to_numpy(self):
        return np.array([i.to_array() for i in self.observations]).flatten()


    def to_tensor(self):
        return torch.FloatTensor(self.to_numpy()).flatten()
    
    def copy(self) -> 'ObservationBuffer':
        return deepcopy(self)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.fwd = nn.Sequential(
            nn.Linear(NN_INPUT_SHAPE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.fwd:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.fwd(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.fwd = nn.Sequential(
            nn.Linear(NN_INPUT_SHAPE + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.fwd:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fwd(x)

class ReplayBuffer:
    def __init__(self, capacity=1_000_000, state_dim=NN_INPUT_SHAPE, device='cpu'):
        self._capacity = capacity
        self._device = device
        
        self._states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self._actions = torch.zeros((capacity, 1), dtype=torch.float32)
        self._rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self._next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self._dones = torch.zeros((capacity, 1), dtype=torch.float32)
        
        self._id = 0
        self._size = 0
    
    def add(self, state, action, reward, next_state, done):
        idx = self._id
        self._states[idx] = torch.as_tensor(state, dtype=torch.float32)
        self._actions[idx, 0] = float(action)
        self._rewards[idx, 0] = float(reward)
        self._next_states[idx] = torch.as_tensor(next_state, dtype=torch.float32)
        self._dones[idx, 0] = float(done)
        
        
        self._id = (self._id + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
 
    def sample(self, batch_size):
        indices = torch.randint(0, self._size, (batch_size,))
        return (
            self._states[indices].to(self._device),
            self._actions[indices].to(self._device),
            self._rewards[indices].to(self._device),
            self._next_states[indices].to(self._device),
            self._dones[indices].to(self._device)
        )
    def __len__(self):
        return self._size


def create_observation(state: State, which_paddle: str,) -> ObservationBuffer:
    if which_paddle == "left":
        paddle_pos = state.left_pad_pos
        paddle_vel = state.left_pad_vel
    else:
        paddle_pos = state.right_pad_pos
        paddle_vel = state.right_pad_vel
    
    return Observation(
        ball_vx=float(state.ball_vel[0]),
        ball_vy=float(state.ball_vel[1]),
        paddle_vy=float(paddle_vel[1]),
        paddle_y=float(paddle_pos[1]),
        ball_x_distance_to_paddle=float(state.ball_pos[0] - paddle_pos[0]),
        ball_y_distance_to_paddle=float(state.ball_pos[1] - paddle_pos[1]),
    )

def train_step(buffer, critic_target, critic_main, actor_main, actor_target, actor_optimizer, critic_optimizer, batch_size=64, gamma=0.98, tau=0.003):
    if len(buffer) < batch_size:
        return None, None
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    with torch.no_grad():
        next_actions = actor_target(next_states)
        target_q = critic_target(next_states, next_actions)
        y = rewards + gamma * target_q * (1 - dones)

    current_q = critic_main(states, actions)
    critic_loss = nn.MSELoss()(current_q, y)
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    actor_actions = actor_main(states)
    actor_loss = -critic_main(states, actor_actions).mean()
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    for target_param, main_param in zip(actor_target.parameters(), actor_main.parameters()):
        target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)
    
    for target_param, main_param in zip(critic_target.parameters(), critic_main.parameters()):
        target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)
    
    return critic_loss.item(), actor_loss.item()

