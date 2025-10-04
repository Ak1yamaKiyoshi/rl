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

DEMO_EACH = 350



HIDDEN_DIM = 64
PREOUTPUT_DIM = 64

from pingpong import State
import numpy as np

def reward_function(state: State):
    # since in replay buffer we capturing not sequences of actions but actions itself... 
    # in needs some constant reward even if no hit. 
    ball_x = state.ball_pos[0] / MAX_X
    reward = 0
    # stay still when possible 
    if 0.5 < ball_x < 1.0 and state.ball_vel[0] < 0 or state.ball_vel[0] > 0: 
        reward += -np.abs(state.right_pad_vel[1]) / MAX_VELOCITY * 2

    # award nearly-hitting the ball. 
    if np.abs(state.left_pad_pos[0] - state.ball_pos[0]) < state.pad_size[0]:
        if state.left_pad_pos[1] + state.pad_size[1]//6  < state.ball_pos[1] <   state.left_pad_pos[1]  + state.pad_size[1] -   state.pad_size[1]//6:
            reward += 15
            reward += np.abs((state.left_pad_vel[1] / MAX_VELOCITY) * 10)


    return  reward 