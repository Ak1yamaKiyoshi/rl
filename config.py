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
PREOUTPUT_DIM = 64

from pingpong import State
import numpy as np

def reward_function(state: State, hit_ball: bool, missed_ball: bool):
    paddle_y_norm = state.left_pad_pos[1] / MAX_Y
    ball_y_norm = state.ball_pos[1] / MAX_Y
    
    x = paddle_y_norm - ball_y_norm
    y_reward = 10 * (-np.abs(np.tanh(x*x) * 6 * np.cos(x))) + 1
    y_reward += state.score * 5
    return  y_reward 