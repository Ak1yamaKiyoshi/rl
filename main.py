import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import cv2
import time 

import torch
from torch import nn
from pid import PIDController

from typing import Tuple

W_WIDTH = 400
W_HEIGHT = 400

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

def process_physics(state: State, dt:float, width, height) -> Tuple[State, bool]:
    new_ball_velocity = state.ball_vel
    new_ball_pos = (state.ball_vel * dt + state.ball_pos)

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

    state.left_pad_pos = new_left_pad_position
    state.left_pad_vel = new_left_pad_vel
    state.right_pad_pos = new_right_pad_position
    state.right_pad_vel = new_right_pad_vel
    
    x_boundary_left = state.left_pad_pos[0] + state.pad_size[0] // 2
    x_boundary_right = state.right_pad_pos[0] - state.pad_size[0] // 2

    left_minmax_y = [
        state.left_pad_pos[1] - state.pad_size[1] // 2,
        state.left_pad_pos[1] + state.pad_size[1] // 2
    ]
    right_minmax_y = [
        state.right_pad_pos[1] - state.pad_size[1] // 2,
        state.right_pad_pos[1] + state.pad_size[1] // 2
    ]

    if new_ball_pos[0] < 0:
        return initialize_state(width, height), True
    elif new_ball_pos[0] > width:
        return initialize_state(width, height), True

    elif new_ball_pos[1] < 0:
        new_ball_pos = new_ball_pos.at[1].set(0)
        new_ball_velocity = new_ball_velocity * jnp.array([1, -1])

    elif new_ball_pos[1] > height:
        new_ball_pos = new_ball_pos.at[1].set(height)
        new_ball_velocity = new_ball_velocity * jnp.array([1, -1])

    if new_ball_pos[0] < x_boundary_left:
        if left_minmax_y[0] < new_ball_pos[1] < left_minmax_y[1]:
            new_ball_pos = new_ball_pos.at[0].set(x_boundary_left + (x_boundary_left - new_ball_pos[0]))
            

            hit_position = (new_ball_pos[1] - state.left_pad_pos[1]) / (state.pad_size[1] / 2)
            hit_position = jnp.clip(hit_position, -0.3, 0.3)
            
            current_speed = jnp.linalg.norm(new_ball_velocity)
            
            base_angle = jnp.pi - jnp.arctan2(new_ball_velocity[1], new_ball_velocity[0])  
            angle_influence = hit_position * (jnp.pi / 6)  
            paddle_influence = state.left_pad_vel[1] * 0.0005

            new_angle = base_angle + angle_influence + paddle_influence

            new_ball_velocity = current_speed * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)])
            state.score += 1

    elif new_ball_pos[0] > x_boundary_right:
        if right_minmax_y[0] < new_ball_pos[1] < right_minmax_y[1]:
            new_ball_pos = new_ball_pos.at[0].set(x_boundary_right - (new_ball_pos[0] - x_boundary_right))
            
            hit_position = (new_ball_pos[1] - state.right_pad_pos[1]) / (state.pad_size[1] / 2)
            hit_position = jnp.clip(hit_position, -0.3, 0.3)
            current_speed = jnp.linalg.norm(new_ball_velocity)

            base_angle = jnp.pi - jnp.arctan2(new_ball_velocity[1], new_ball_velocity[0]) 
            angle_influence = hit_position * (jnp.pi / 6)  
            paddle_influence = state.right_pad_vel[1] * 0.0005

            new_angle = base_angle + angle_influence + paddle_influence
            
            new_ball_velocity = current_speed * jnp.array([jnp.cos(new_angle), jnp.sin(new_angle)])
            state.score += 1

    state.ball_pos = new_ball_pos
    state.ball_vel = new_ball_velocity
    return state, False
    

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

NN_OBSERVATIONS = 10

def update_policy_observations(state:State, n=NN_OBSERVATIONS) -> State:
    observation = [
        state.ball_pos[0], state.ball_pos[1], 
        state.ball_vel[0], state.ball_vel[1], 
        state.left_pad_pos[0], state.left_pad_pos[1],
        state.left_pad_vel[0], state.left_pad_vel[1],
    ]
    state.observations.append(observation)
    while len(state.observations) > n:
        state.observations.pop(0)
    return state


class PongModel(nn.Module):
    def __init__(self):
        self.fwd = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(NN_OBSERVATIONS*8, 8),
            nn.Dropout1d(),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.fwd(x)




if __name__ == "__main__":
    target_fps = 60
    target_dt = 1.0 / target_fps
    states = [ initialize_state(W_WIDTH, W_HEIGHT) for i in range(40)]

    for i in range(400):
        for j, state in enumerate(states):    
            update_pid(state)
            state, restarted = process_physics(state, target_dt, W_WIDTH, W_HEIGHT)
            if restarted:
                print(f" {j:03d} restarted")
