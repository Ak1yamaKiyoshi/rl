import numpy as np
from dataclasses import dataclass
import cv2
import time 

import torch
from torch import nn
from pid import PIDController

import random
from typing import Tuple



W_WIDTH = 900
W_HEIGHT = 600
NN_HZ = 50
NN_DT = 1 / NN_HZ
NN_OBSERVATIONS = 10
NN_INPUT_SHAPE = 6

MAX_VELOCITY = 500
MAX_Y = W_HEIGHT
MAX_X = W_WIDTH
PADDLE_VELOCITY_MAX = MAX_VELOCITY

DEMO_EACH = 300

HIDDEN_DIM = 64
PREOUTPUT_DIM = 32

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


@dataclass
class State:
    ball_pos: np.ndarray # center
    ball_vel: np.ndarray
    
    left_pad_pos: np.ndarray # center
    left_pad_vel: np.ndarray

    right_pad_pos: np.ndarray # center
    right_pad_vel: np.ndarray

    score: int
    pad_size: np.ndarray 
    pid_controller: PIDController
    observations = []

def initialize_state(width, height) -> State:
    pads_y = height // 2 
    left_pad_x = 10 
    right_pad_x = width - 10 
    ball_pos = np.array([width//2, height//2], dtype=np.float32)

    speed = 350
    angle = (np.random.random() - 0.5) * (np.pi/2)       
    if np.random.random() > 0.5:
        angle = np.pi - angle  
    ball_vel = speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    return State(
        ball_pos=ball_pos, 
        ball_vel=ball_vel, 
        left_pad_pos=np.array([left_pad_x, pads_y], dtype=np.float32), 
        left_pad_vel=np.array([0.0, 0.0], dtype=np.float32), 
        right_pad_pos=np.array([right_pad_x, pads_y], dtype=np.float32),
        right_pad_vel=np.array([0.0, 0.0], dtype=np.float32), 
        pad_size=np.array([10, 50], dtype=np.float32),
        score=0,
        pid_controller=PIDController(0.1, 0.05, 0.0)
    )
def process_physics(state: State, dt: float, width, height) -> Tuple[State, bool]:
    new_ball_pos = state.ball_pos + state.ball_vel * dt
    new_left_pad_position = state.left_pad_pos + state.left_pad_vel * dt
    new_right_pad_position = state.right_pad_pos + state.right_pad_vel * dt

    new_left_pad_vel = state.left_pad_vel.copy()
    new_right_pad_vel = state.right_pad_vel.copy()

    left_pad_top = new_left_pad_position[1] - state.pad_size[1] // 2
    left_pad_bottom = new_left_pad_position[1] + state.pad_size[1] // 2
    
    if left_pad_top < 0:
        new_left_pad_position[1] = state.pad_size[1] // 2
        new_left_pad_vel[1] = 0
    elif left_pad_bottom > height:
        new_left_pad_position[1] = height - state.pad_size[1] // 2
        new_left_pad_vel[1] = 0

    right_pad_top = new_right_pad_position[1] - state.pad_size[1] // 2
    right_pad_bottom = new_right_pad_position[1] + state.pad_size[1] // 2
    
    if right_pad_top < 0:
        new_right_pad_position[1] = state.pad_size[1] // 2
        new_right_pad_vel[1] = 0
    elif right_pad_bottom > height:
        new_right_pad_position[1] = height - state.pad_size[1] // 2
        new_right_pad_vel[1] = 0

    x_boundary_left = new_left_pad_position[0] + state.pad_size[0] // 2
    x_boundary_right = new_right_pad_position[0] - state.pad_size[0] // 2

    left_minmax_y = np.array([
        new_left_pad_position[1] - state.pad_size[1] // 2,
        new_left_pad_position[1] + state.pad_size[1] // 2
    ])
    right_minmax_y = np.array([
        new_right_pad_position[1] - state.pad_size[1] // 2,
        new_right_pad_position[1] + state.pad_size[1] // 2
    ])

    old_ball_x = state.ball_pos[0]
    old_ball_y = state.ball_pos[1]
    
    new_ball_velocity = state.ball_vel.copy()
    new_score = state.score
    ball_reflected = False
    original_speed = np.linalg.norm(state.ball_vel)
    
    if new_ball_pos[1] < 0:
        new_ball_pos[1] = -new_ball_pos[1]
        new_ball_velocity[1] = -new_ball_velocity[1]
    elif new_ball_pos[1] > height:
        new_ball_pos[1] = 2 * height - new_ball_pos[1]
        new_ball_velocity[1] = -new_ball_velocity[1]

    if new_ball_velocity[0] < 0:
        crossed_left = (old_ball_x >= x_boundary_left) and (new_ball_pos[0] <= x_boundary_left)
        
        if crossed_left and np.abs(new_ball_pos[0] - old_ball_x) > 1e-6:
            t_left = (x_boundary_left - old_ball_x) / (new_ball_pos[0] - old_ball_x)
            t_left = float(np.clip(t_left, 0, 1))
            
            intersection_y_left = old_ball_y + t_left * (new_ball_pos[1] - old_ball_y)
            
            hit_left_paddle = (intersection_y_left >= left_minmax_y[0]) and (intersection_y_left <= left_minmax_y[1])
            
            if hit_left_paddle:
                ball_reflected = True
                
                hit_position_left = float((intersection_y_left - new_left_pad_position[1]) / (state.pad_size[1] / 2))
                hit_position_left = np.clip(hit_position_left, -1.0, 1.0)
                
                new_vx = -new_ball_velocity[0]
                
                angle_deflection = hit_position_left * (np.pi / 6)
                
                paddle_influence = float(new_left_pad_vel[1]) * 0.3
                
                new_vy = new_ball_velocity[1] + np.tan(angle_deflection) * np.abs(new_vx) + paddle_influence
                
                new_ball_velocity = np.array([new_vx, new_vy], dtype=np.float32)
                
                current_speed = np.linalg.norm(new_ball_velocity)
                if current_speed > 0:
                    new_ball_velocity = new_ball_velocity * (original_speed / current_speed)
                
                remaining_time = 1.0 - t_left
                new_ball_pos = np.array([x_boundary_left, intersection_y_left], dtype=np.float32) + new_ball_velocity * dt * remaining_time
                
                new_score = new_score + 1

    if new_ball_velocity[0] > 0 and not ball_reflected:
        crossed_right = (old_ball_x <= x_boundary_right) and (new_ball_pos[0] >= x_boundary_right)
        
        if crossed_right and np.abs(new_ball_pos[0] - old_ball_x) > 1e-6:
            t_right = (x_boundary_right - old_ball_x) / (new_ball_pos[0] - old_ball_x)
            t_right = float(np.clip(t_right, 0, 1))
            
            intersection_y_right = old_ball_y + t_right * (new_ball_pos[1] - old_ball_y)
            
            hit_right_paddle = (intersection_y_right >= right_minmax_y[0]) and (intersection_y_right <= right_minmax_y[1])
            
            if hit_right_paddle:
                ball_reflected = True
                
                hit_position_right = float((intersection_y_right - new_right_pad_position[1]) / (state.pad_size[1] / 2))
                hit_position_right = np.clip(hit_position_right, -1.0, 1.0)
                
                new_vx = -new_ball_velocity[0]
                
                angle_deflection = hit_position_right * (np.pi / 6)
                
                paddle_influence = float(new_right_pad_vel[1]) * 0.3
                
                new_vy = new_ball_velocity[1] + np.tan(angle_deflection) * np.abs(new_vx) + paddle_influence
                
                new_ball_velocity = np.array([new_vx, new_vy], dtype=np.float32)
                
                current_speed = np.linalg.norm(new_ball_velocity)
                if current_speed > 0:
                    new_ball_velocity = new_ball_velocity * (original_speed / current_speed)
                
                remaining_time = 1.0 - t_right
                new_ball_pos = np.array([x_boundary_right, intersection_y_right], dtype=np.float32) + new_ball_velocity * dt * remaining_time
                
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
    )
    
    return new_state, False

def render(height, width, state:State):
    canvas = np.zeros([width, height, 3])
    cv2.circle(canvas, list(map(int, state.ball_pos.tolist())), 5, (255 ,255, 255))
    cv2.putText(canvas, str(state.score), [width//2, height//3], 0, 1, (255, 255, 255) )

    for pad in [state.left_pad_pos, state.right_pad_pos]:
        x, y = pad
        pt1 = int(x-state.pad_size[0]//2), int(y-state.pad_size[1]//2)
        pt2 = int(x+state.pad_size[0]//2), int(y+state.pad_size[1]//2)
        cv2.rectangle(canvas, pt1, pt2, (255, 255, 255, -1))
    return canvas

def update_pid(state:State, dt=NN_DT):
    measurement = -(state.ball_pos[1] - state.right_pad_pos[1])
    ctl = state.pid_controller.update(measurement, dt)
    state.right_pad_vel[1] = np.clip(ctl * 100, -1000, 1000)
    return state

def reward_function(state: State, hit_ball: bool, missed_ball: bool):
    paddle_y_norm = state.left_pad_pos[1] / MAX_Y
    ball_y_norm = state.ball_pos[1] / MAX_Y
    
    x = paddle_y_norm - ball_y_norm
    y_reward = 10 * (-np.abs(np.tanh(x*x) * 6 * np.cos(x))) + 1
    y_reward += state.score * 5


    return  y_reward 

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

def train_step(buffer, batch_size=64, gamma=0.99, tau=0.005):
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


if __name__ == "__main__":     
    try:
        torch.set_num_threads(12)
        torch.set_flush_denormal(True)
        
        checkpoints_file = open("checkpoint_meta", "w+") 

        torch.serialization.add_safe_globals([
            Actor, 
            Critic,
            nn.Sequential,
            nn.Linear,
            nn.ReLU,
            nn.Tanh
        ])

        actor_main = Actor()
        # torch.load('checkpoints/0000038798_actor_0.08564683049917221.pt', weights_only=False, map_location='cpu')
        actor_target = Actor()
        actor_target.load_state_dict(actor_main.state_dict())
        
        critic_main = Critic()
        #torch.load('checkpoints/0000038798_critic_0.08564683049917221.pt', weights_only=False, map_location='cpu')
        critic_target = Critic()
        critic_target.load_state_dict(critic_main.state_dict())


        actor_optimizer = torch.optim.Adam(actor_main.parameters(), lr=1e-3, fused=False)
        critic_optimizer = torch.optim.Adam(critic_main.parameters(), lr=1e-3, fused=False)

        actor_target.eval()
        critic_target.eval()

        for param in actor_target.parameters():
            param.requires_grad = False
        for param in critic_target.parameters():
            param.requires_grad = False

        buffer = ReplayBuffer(capacity=1_000_000)
        state = initialize_state(W_WIDTH, W_HEIGHT)

        episode_reward = 0
        noise_scale = 0.05


        old_score = 0
        restart_amount = 0


        step_in_observation = 0
        for step in range(1_000_00_000):
            step_in_observation += 1
            obs = create_observation(state, "left")
            obs_tensor = torch.FloatTensor(obs.to_array()).unsqueeze(0)

            with torch.no_grad():
                action = actor_main(obs_tensor).item() + np.random.random(1)[0] * noise_scale

            state.left_pad_vel[1] = action * PADDLE_VELOCITY_MAX
            next_state, done = process_physics(state, NN_DT, W_WIDTH, W_HEIGHT)

            if not done:
                hit_ball = (next_state.score > old_score)
                old_score = next_state.score
                
                next_obs = create_observation(next_state, "left")
                
                reward = reward_function(state, hit_ball, done)
                episode_reward += reward
                
                buffer.add(obs.to_array(), action, reward, next_obs.to_array(), float(done))
                
                if restart_amount % 8 == 0: 
                    losses = train_step(buffer, batch_size=128, )
                    if losses[0] is not None and step % 100 == 0:
                        print(f"Step {step}, Critic Loss: {losses[0]:.3f}, Actor Loss: {losses[1]:.3f}")

                next_state = update_pid(next_state)
                state = next_state

            if state.score >= 50 or  step_in_observation >= 10_000:
                done = True


            if done:
                step_in_observation = 0
                print(f"[{restart_amount % DEMO_EACH:3d}/{DEMO_EACH}] reward: {reward:8.6f}, score: {state.score:2d}")
                restart_amount += 1
                state = initialize_state(W_WIDTH, W_HEIGHT)
                episode_reward = 0
                old_score = 0

                if restart_amount % DEMO_EACH == 0:
                    checkpoints_file.write(str({
                        "actor_loss": losses[1],
                        "critic_loss": losses[0], 
                        "reward": episode_reward,
                    }))

                    torch.save(actor_main, f"checkpoints/{step:010d}_actor_{losses[0]}.pt")
                    torch.save(critic_main, f"checkpoints/{step:010d}_critic_{losses[0]}.pt")
        
                    actor_main.eval()


                    for _ in range(4):
                        done_demo = False
                        while not done_demo:
                            state, done_demo = process_physics(state, NN_DT, W_WIDTH, W_HEIGHT)
                            obs = create_observation(state, "left")
                            obs_tensor = torch.FloatTensor(obs.to_array()).unsqueeze(0)
                            action = actor_main(obs_tensor).item()

                            state.left_pad_vel[1] = action * PADDLE_VELOCITY_MAX
                            state = update_pid(state)
                            
                            reward_ = reward_function(state, False, False)

                            img = render(W_WIDTH, W_HEIGHT, state)
                            overlay = img.copy()
                            cv2.rectangle(overlay, (0, 0), (W_WIDTH, 100), (0, 0, 0), -1)
                            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
                            
                            cv2.putText(img, f"DEMO MODE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            cv2.putText(img, f"Action: {action:+.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                            cv2.putText(img, f"Reward: {reward_}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            cv2.imshow("Training", img)
                            cv2.waitKey(12)


                    cv2.destroyAllWindows()
                    actor_main.train()
                    state = initialize_state(W_WIDTH, W_HEIGHT)


            noise_scale *= 0.9995
    except KeyboardInterrupt:
        pass