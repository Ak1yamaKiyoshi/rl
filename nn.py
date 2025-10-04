import numpy as np
from dataclasses import dataclass
from config import *

import torch
from torch import nn

def add_noise(action, noise_scale):
    noise = np.random.normal(0, noise_scale)
    return np.clip(action + noise, -PADDLE_VELOCITY_MAX, PADDLE_VELOCITY_MAX)

@dataclass 
class ObservationBuffer:
    ball_vx: float          
    ball_vy: float          

    paddle_vy: float
    paddle_y: float

    ball_x_distance_to_paddle: float
    ball_y_distance_to_paddle: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.ball_vx / MAX_VELOCITY,
            self.ball_vy / MAX_VELOCITY,
            self.paddle_vy / MAX_VELOCITY,
            self.paddle_y / MAX_Y,

            self.ball_x_distance_to_paddle / MAX_X,
            self.ball_y_distance_to_paddle / MAX_Y,
        ])

class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.fwd = nn.Sequential(
            nn.Linear(NN_INPUT_SHAPE, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, PREOUTPUT_DIM),
            nn.ReLU(),
            nn.Linear(PREOUTPUT_DIM, 1),
            nn.Tanh() # -1 ... 1 * BASE CTL - would be actor ctl 
        )

    def forward(self, x):
        return self.fwd(x)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.fwd = nn.Sequential(
            nn.Linear(NN_INPUT_SHAPE + 1, HIDDEN_DIM), # state + action 
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, PREOUTPUT_DIM),
            nn.ReLU(),
            nn.Linear(PREOUTPUT_DIM, 1), 
            # Q value 
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fwd(x)


class ReplayBuffer:
    def __init__(self, capacity=1_000_000, state_dim=6, device='cpu'):
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
        self._actions[idx, 0] = float(action)  # Convert to float
        self._rewards[idx, 0] = float(reward)  # Convert to float
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



def create_observation(state: State, which_paddle: str) -> ObservationBuffer:
    if which_paddle == "left":
        paddle_pos = state.left_pad_pos
        paddle_vel = state.left_pad_vel
    else:
        paddle_pos = state.right_pad_pos
        paddle_vel = state.right_pad_vel
    
    return ObservationBuffer(
        ball_vx=float(state.ball_vel[0]),
        ball_vy=float(state.ball_vel[1]),
        paddle_vy=float(paddle_vel[1]),
        paddle_y=float(paddle_pos[1]),
        ball_x_distance_to_paddle=float(state.ball_pos[0] - paddle_pos[0]),
        ball_y_distance_to_paddle=float(state.ball_pos[1] - paddle_pos[1])
    )

def train_step(buffer, critic_target, critic_main, actor_main, actor_target, actor_optimizer, critic_optimizer, batch_size=64, gamma=0.99, tau=0.005):
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

