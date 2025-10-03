import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import cv2
import time 

import torch
from torch import nn
from pid import PIDController

import random
from typing import Tuple



W_WIDTH = 400
W_HEIGHT = 400
NN_HZ = 10
NN_DT = 1 / NN_HZ
NN_OBSERVATIONS = 10
NN_INPUT_SHAPE = 6

MAX_VELOCITY = 500 
MAX_Y = W_HEIGHT
MAX_X = W_WIDTH
PADDLE_VELOCITY_MAX = MAX_VELOCITY

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

    def to_array(self) -> jnp.ndarray:
        return jnp.array([
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
            nn.Linear(NN_INPUT_SHAPE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh() # -1 ... 1 * BASE CTL - would be actor ctl 
        )

    def forward(self, x):
        return self.fwd(x) * PADDLE_VELOCITY_MAX

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.fwd = nn.Sequential(
            nn.Linear(NN_INPUT_SHAPE + 1, 128), # state + action 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1), 
            # Q value 
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fwd(x)

class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self._capacity = capacity

        self._states = [None for i in range(capacity)]
        self._actions = [None for i in range(capacity)]
        self._rewards = [None for i in range(capacity)]
        self._next_states = [None for i in range(capacity)]
        self._dones = [None for i in range(capacity)]

        self._id = 0
        self._size = 0

    def add(self, state, action, reward, next_state, done):
        self._states[self._id] = state
        self._actions[self._id] = action
        self._rewards[self._id] = reward
        self._next_states[self._id] = next_state
        self._dones[self._id] = done
        
        self._id = (self._id + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size):
        indices = [random.randint(0, self._size - 1) for _ in range(batch_size)]
        
        states = [self._states[i] for i in indices]
        actions = [self._actions[i] for i in indices]
        rewards = [self._rewards[i] for i in indices]
        next_states = [self._next_states[i] for i in indices]
        dones = [self._dones[i] for i in indices]
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)).unsqueeze(1),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

    def __len__(self):
        return self._size

@dataclass
class State:
    ball_pos: jnp.ndarray # center
    ball_vel: jnp.ndarray
    
    left_pad_pos: jnp.ndarray # center
    left_pad_vel: jnp.ndarray

    right_pad_pos: jnp.ndarray # center
    right_pad_vel: jnp.ndarray

    score: int
    pad_size: jnp.ndarray 
    pid_controller: PIDController
    observations = []

def initialize_state(width, height) -> State:
    pads_y = height // 2 
    left_pad_x = 10 
    right_pad_x = width - 10 
    ball_pos = jnp.array([width//2, height//2])

    speed = 350
    angle = (np.random.random() -0.5 )* (np.pi/2)       
    if np.random.random() > 0.5:
        angle = np.pi - angle  
    ball_vel = speed * jnp.array([np.cos(angle), np.sin(angle)])

    return State(
        ball_pos=ball_pos, 
        ball_vel=ball_vel, 
        left_pad_pos=jnp.array([left_pad_x, pads_y]), 
        left_pad_vel=jnp.array([0.0, 0.0]), 
        right_pad_pos=jnp.array([right_pad_x, pads_y]),
        right_pad_vel=jnp.array([0.0, 0.0]), 
        pad_size=jnp.array([10, 50]),
        score=0,
        pid_controller=PIDController(0.05, 0.3, 0.0)
    )

def process_physics(state: State, dt: float, width, height) -> Tuple[State, bool]:
    new_ball_pos = state.ball_pos + state.ball_vel * dt
    new_left_pad_position = state.left_pad_pos + state.left_pad_vel * dt
    new_right_pad_position = state.right_pad_pos + state.right_pad_vel * dt

    new_left_pad_vel = state.left_pad_vel
    new_right_pad_vel = state.right_pad_vel

    left_pad_top = new_left_pad_position[1] - state.pad_size[1] // 2
    left_pad_bottom = new_left_pad_position[1] + state.pad_size[1] // 2
    
    if left_pad_top < 0:
        new_left_pad_position = new_left_pad_position.at[1].set(state.pad_size[1] // 2)
        new_left_pad_vel = new_left_pad_vel.at[1].set(0)
    elif left_pad_bottom > height:
        new_left_pad_position = new_left_pad_position.at[1].set(height - state.pad_size[1] // 2)
        new_left_pad_vel = new_left_pad_vel.at[1].set(0)

    right_pad_top = new_right_pad_position[1] - state.pad_size[1] // 2
    right_pad_bottom = new_right_pad_position[1] + state.pad_size[1] // 2
    
    if right_pad_top < 0:
        new_right_pad_position = new_right_pad_position.at[1].set(state.pad_size[1] // 2)
        new_right_pad_vel = new_right_pad_vel.at[1].set(0)
    elif right_pad_bottom > height:
        new_right_pad_position = new_right_pad_position.at[1].set(height - state.pad_size[1] // 2)
        new_right_pad_vel = new_right_pad_vel.at[1].set(0)

    x_boundary_left = new_left_pad_position[0] + state.pad_size[0] // 2
    x_boundary_right = new_right_pad_position[0] - state.pad_size[0] // 2

    left_minmax_y = jnp.array([
        new_left_pad_position[1] - state.pad_size[1] // 2,
        new_left_pad_position[1] + state.pad_size[1] // 2
    ])
    right_minmax_y = jnp.array([
        new_right_pad_position[1] - state.pad_size[1] // 2,
        new_right_pad_position[1] + state.pad_size[1] // 2
    ])

    old_ball_x = state.ball_pos[0]
    old_ball_y = state.ball_pos[1]
    
    new_ball_velocity = state.ball_vel
    new_score = state.score
    ball_reflected = False
    
    if new_ball_pos[1] < 0:
        new_ball_pos = new_ball_pos.at[1].set(-new_ball_pos[1])
        new_ball_velocity = new_ball_velocity.at[1].set(-new_ball_velocity[1])
    elif new_ball_pos[1] > height:
        new_ball_pos = new_ball_pos.at[1].set(2 * height - new_ball_pos[1])
        new_ball_velocity = new_ball_velocity.at[1].set(-new_ball_velocity[1])

    if new_ball_velocity[0] < 0:
        crossed_left = (old_ball_x >= x_boundary_left) and (new_ball_pos[0] <= x_boundary_left)
        
        if crossed_left and jnp.abs(new_ball_pos[0] - old_ball_x) > 1e-6:
            t_left = (x_boundary_left - old_ball_x) / (new_ball_pos[0] - old_ball_x)
            t_left = float(jnp.clip(t_left, 0, 1))
            
            intersection_y_left = old_ball_y + t_left * (new_ball_pos[1] - old_ball_y)
            
            hit_left_paddle = (intersection_y_left >= left_minmax_y[0]) and (intersection_y_left <= left_minmax_y[1])
            
            if hit_left_paddle:
                ball_reflected = True
                
                hit_position_left = float((intersection_y_left - new_left_pad_position[1]) / (state.pad_size[1] / 2))
                hit_position_left = jnp.clip(hit_position_left, -1.0, 1.0)
                
                new_vx = -new_ball_velocity[0]
                
                angle_deflection = hit_position_left * (jnp.pi / 6)  # Max ±30 degrees
                
                paddle_influence = float(new_left_pad_vel[1]) * 0.3
                
                new_vy = new_ball_velocity[1] + jnp.tan(angle_deflection) * jnp.abs(new_vx) + paddle_influence
                
                new_ball_velocity = jnp.array([new_vx, new_vy])
                
                remaining_time = 1.0 - t_left
                new_ball_pos = jnp.array([x_boundary_left, intersection_y_left]) + new_ball_velocity * dt * remaining_time
                
                new_score = new_score + 1

    # Right paddle collision (ball moving right: vx > 0)
    if new_ball_velocity[0] > 0 and not ball_reflected:
        crossed_right = (old_ball_x <= x_boundary_right) and (new_ball_pos[0] >= x_boundary_right)
        
        if crossed_right and jnp.abs(new_ball_pos[0] - old_ball_x) > 1e-6:
            t_right = (x_boundary_right - old_ball_x) / (new_ball_pos[0] - old_ball_x)
            t_right = float(jnp.clip(t_right, 0, 1))
            
            intersection_y_right = old_ball_y + t_right * (new_ball_pos[1] - old_ball_y)
            
            hit_right_paddle = (intersection_y_right >= right_minmax_y[0]) and (intersection_y_right <= right_minmax_y[1])
            
            if hit_right_paddle:
                ball_reflected = True
                
                hit_position_right = float((intersection_y_right - new_right_pad_position[1]) / (state.pad_size[1] / 2))
                hit_position_right = jnp.clip(hit_position_right, -1.0, 1.0)
                
                new_vx = -new_ball_velocity[0]
                
                angle_deflection = hit_position_right * (jnp.pi / 6)  # Max ±30 degrees
                
                paddle_influence = float(new_right_pad_vel[1]) * 0.3
                
                new_vy = new_ball_velocity[1] + jnp.tan(angle_deflection) * jnp.abs(new_vx) + paddle_influence
                
                new_ball_velocity = jnp.array([new_vx, new_vy])
                
                remaining_time = 1.0 - t_right
                new_ball_pos = jnp.array([x_boundary_right, intersection_y_right]) + new_ball_velocity * dt * remaining_time
                
                new_score = new_score + 1
    
    scored = (new_ball_pos[0] < 0) or (new_ball_pos[0] > width)
    
    if scored:
        return initialize_state(width, height), True

    new_state = State(
        ball_pos=new_ball_pos,
        ball_vel=new_ball_velocity,
        left_pad_pos=new_left_pad_position,
        left_pad_vel=new_left_pad_vel,
        right_pad_pos=new_right_pad_position,
        right_pad_vel=new_right_pad_vel,
        score=new_score,
        pad_size=state.pad_size,
        pid_controller=state.pid_controller,
        #observations=state.observations
    )
    
    return new_state, False

def render(width, height, state:State):
    canvas = np.zeros([width, height, 3])
    cv2.circle(canvas, list(map(int, state.ball_pos.tolist())), 5, (255 ,255, 255))
    cv2.putText(canvas, str(state.score), [width//2, height//3], 0, 1, (255, 255, 255) )

    for pad in [state.left_pad_pos, state.right_pad_pos]:
        x, y = pad
        pt1 = int(x-state.pad_size[0]//2), int(y-state.pad_size[1]//2)
        pt2 = int(x+state.pad_size[0]//2), int(y+state.pad_size[1]//2)
        cv2.rectangle(canvas, pt1, pt2, (255, 255, 255, -1))
    return canvas

def update_pid(state:State):
    measurement = -(state.ball_pos[1] - state.right_pad_pos[1])
    ctl = state.pid_controller.update(measurement, target_dt)
    state.right_pad_vel = state.right_pad_vel.at[1].set(np.clip(ctl * 100, -1000, 1000))
    return state

def reward_function(state: State, hit_ball: bool, missed_ball: bool):
    if hit_ball:
        return 50.0 
    if missed_ball:
        return -50.0
    
    paddle_y_norm = state.left_pad_pos[1] / MAX_Y
    ball_y_norm = state.ball_pos[1] / MAX_Y
    
    y_error = paddle_y_norm - ball_y_norm
    y_reward = (-np.sqrt(np.abs(y_error * 3)) + 0.8) * 15
    
    boundaries_reward = (np.sin(np.pi * paddle_y_norm) - 0.5) * 5
    
    return boundaries_reward + y_reward

actor_main = Actor()
actor_target = Actor()
actor_target.load_state_dict(actor_main.state_dict())

critic_main = Critic()
critic_target = Critic()
critic_target.load_state_dict(critic_main.state_dict())

actor_optimizer = torch.optim.Adam(actor_main.parameters(), lr=1e-4)
critic_optimizer = torch.optim.Adam(critic_main.parameters(), lr=1e-3)

actor_target.eval()
critic_target.eval()

for param in actor_target.parameters():
    param.requires_grad = False
for param in critic_target.parameters():
    param.requires_grad = False



# if __name__ == "__main__":
#     target_fps = NN_HZ
#     target_dt = NN_DT
#     state = initialize_state(W_WIDTH, W_HEIGHT)
#     for i in range(400):
#         update_pid(state)
#         state, restarted = process_physics(state, target_dt, W_WIDTH, W_HEIGHT)
#         img = render(W_WIDTH, W_HEIGHT, state)
#         cv2.imshow("f", img)
#         cv2.waitKey(0)
    
