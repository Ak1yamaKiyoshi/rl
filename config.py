W_WIDTH = 900
W_HEIGHT = 500

NN_HZ = 40
NN_DT = 1 / NN_HZ
NN_DATA_PER_SAMPLE = 6
NN_OBSERVATIONS = 40
NN_INPUT_SHAPE = NN_DATA_PER_SAMPLE * NN_OBSERVATIONS

MAX_VELOCITY = 500
MAX_Y = W_HEIGHT
MAX_X = W_WIDTH
PADDLE_VELOCITY_MAX = MAX_VELOCITY


HIDDEN_DIM = 64
PREOUTPUT_DIM = 64

DEMO_EACH = 500
EVAL_EACH =50
TRAIN_EVERY = 500
TRAIN_BATCH_SIZE = 128


EXPNAME = "noneednomove-v2"


def reward_function(state):
    ball_x = state.ball_pos[0]
    ball_y = state.ball_pos[1]
    ball_vx = state.ball_vel[0]
    
    pad_x = state.left_pad_pos[0]
    pad_y = state.left_pad_pos[1]
    pad_vy = state.left_pad_vel[1]
    pad_height = state.pad_size[1]
    pad_width = state.pad_size[0]
    
    paddle_speed_norm = abs(pad_vy) / MAX_VELOCITY if MAX_VELOCITY > 0 else 0
    
    reward = 0
    
    if ball_vx > 0:
        reward -= paddle_speed_norm * 10
    else:
        reward -= paddle_speed_norm * 3
    
    x_boundary = pad_x + pad_width / 2
    one_pad_away_x = x_boundary + pad_width
    
    if x_boundary <= ball_x <= one_pad_away_x and ball_vx < 0:
        y_distance = abs(ball_y - pad_y)
        quarter_pad = pad_height / 4
        half_pad = pad_height / 2
        
        if y_distance < pad_height / 8:
            reward += 200
            reward += paddle_speed_norm * 20
        elif y_distance < quarter_pad:
            reward += 30
            reward += paddle_speed_norm * 15
        elif y_distance < half_pad:
            reward += 20
            reward += paddle_speed_norm * 10
        elif y_distance < pad_height:
            reward += 10
            reward += paddle_speed_norm * 5

    if ball_x < x_boundary:
        reward -= 100
    
    if ball_vx > 0:
        center_y = MAX_Y / 2
        distance_from_center = abs(pad_y - center_y)
        center_norm = 1 - (distance_from_center / (MAX_Y / 2))
        reward += center_norm * 5


    return reward