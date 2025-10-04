from main import Actor, Critic, State, render, initialize_state, process_physics, W_WIDTH, W_HEIGHT, NN_DT, PADDLE_VELOCITY_MAX, update_pid, reward_function, create_observation, NN_HZ
import cv2

import time
import torch
from torch import nn
import numpy as np
torch.serialization.add_safe_globals([
    Actor, 
    Critic,
    nn.Sequential,
    nn.Linear,
    nn.ReLU,
    nn.Tanh
])

def render(height, width, state:State):
    canvas = np.zeros([width, height, 3])
    cv2.circle(canvas, list(map(int, state.ball_pos.tolist())), 5, (255, 255, 255), -1)
    cv2.putText(canvas, str(state.score), [width//2, height//3], 0, 1, (255, 255, 255))
    x, y = state.left_pad_pos
    pt1 = int(x - state.pad_size[0]//2), int(y - state.pad_size[1]//2)
    pt2 = int(x + state.pad_size[0]//2), int(y + state.pad_size[1]//2)
    cv2.rectangle(canvas, pt1, pt2, (0, 255, 0), -1)

    x, y = state.right_pad_pos
    pt1 = int(x - state.pad_size[0]//2), int(y - state.pad_size[1]//2)
    pt2 = int(x + state.pad_size[0]//2), int(y + state.pad_size[1]//2)
    cv2.rectangle(canvas, pt1, pt2, (255, 0, 0), -1)

    cv2.putText(canvas, "Actor Policy", [10, 20], 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(canvas, "PID Controller", [width - 130, 20], 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return canvas


actor_main = torch.load("checkpoints/actor_15.875.pt", weights_only=False)
actor_main.eval()

state = initialize_state(W_WIDTH, W_HEIGHT)
done_demo = False

TARGET_FPS = NN_HZ
FRAME_TIME = 1.0 / TARGET_FPS  

while not done_demo:
    frame_start = time.time()
    
    state, done_demo = process_physics(state, NN_DT, W_WIDTH, W_HEIGHT)
    obs = create_observation(state, "left")
    obs_tensor = torch.FloatTensor(obs.to_array()).unsqueeze(0)
    action = actor_main(obs_tensor).item()

    state.left_pad_vel[1] = action * PADDLE_VELOCITY_MAX
    state = update_pid(state)
    
    reward_ = reward_function(state)

    img = render(W_WIDTH, W_HEIGHT, state)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W_WIDTH, 100), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    cv2.putText(img, f"{action:+.1f}", (int(state.left_pad_pos[0]), int(state.left_pad_pos[1]) - int(state.pad_size[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    
    cv2.imshow('Pong', img)

    elapsed = time.time() - frame_start
    wait_time = max(1, int((FRAME_TIME - elapsed) * 1000))  # Convert to ms, minimum 1ms
    
    # Wait and check for exit key
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()




# 
# use previous states in model (with its actions )
# use threads to fill replay bffer 