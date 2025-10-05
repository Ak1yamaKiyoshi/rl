W_WIDTH = 900
W_HEIGHT = 600

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

DEMO_EACH = 100
EVAL_EACH = 100
TRAIN_EVERY = 2
TRAIN_BATCH_SIZE = 128


EXPNAME = "noneednomove"

def reward_function(state):
    reward = 0

    ball_x_normalized = state.ball_pos[0] / MAX_X
    ball_y = state.ball_pos[1]
    pad_y = state.left_pad_pos[1]
    pad_height = state.pad_size[1]
    pad_width = state.pad_size[0]

    y_distance = abs(ball_y - pad_y)
    y_distance_normalized = y_distance / MAX_Y

    pad_top = pad_y - pad_height / 2
    pad_bottom = pad_y + pad_height / 2
    ball_in_pad_range = pad_top <= ball_y <= pad_bottom

    ball_x = state.ball_pos[0]
    pad_x = state.left_pad_pos[0]
    x_distance = abs(ball_x - pad_x)
    ball_near_paddle = x_distance < (pad_width / 2 + 20)

    paddle_speed = abs(state.left_pad_vel[1])
    paddle_speed_normalized = paddle_speed / MAX_VELOCITY if MAX_VELOCITY > 0 else 0
    ball_vy = state.ball_vel[1]
    pad_vy = state.left_pad_vel[1]

    center_y = MAX_Y / 2
    distance_from_center = abs(pad_y - center_y)
    distance_from_center_normalized = distance_from_center / (MAX_Y / 2)

    well_positioned = y_distance < pad_height * 1.5

    if state.ball_vel[0] < 0 and ball_x_normalized < 0.3 and ball_in_pad_range:
        reward += 5 

        center_distance = abs(ball_y - pad_y)
        quarter_pad = pad_height / 4

        if center_distance < quarter_pad:
            if center_distance < pad_height / 16:
                reward += 15
            else:
                reward += 10
        
        if well_positioned:
            reward -= paddle_speed_normalized * 12

    if state.ball_vel[0] < 0 and ball_x_normalized < 0.3 and not ball_in_pad_range:
        reward -= y_distance_normalized * 10
        
        if not well_positioned:
            reward -= paddle_speed_normalized * 2
        else:
            reward -= paddle_speed_normalized * 15

    if state.ball_vel[0] > 0:
        reward -= paddle_speed_normalized * 20

        center_bonus = (1 - distance_from_center_normalized) * 3
        reward += center_bonus
        
        if distance_from_center_normalized > 0.3:
            reward -= distance_from_center_normalized * 5

    if state.ball_vel[0] < 0 and ball_x_normalized < 0.2:
        reward -= paddle_speed_normalized * 5

    if state.ball_vel[0] < 0 and ball_x_normalized >= 0.3:
        if well_positioned:
            reward -= paddle_speed_normalized * 18
        else:
            reward -= paddle_speed_normalized * 10

    if ball_in_pad_range and ball_near_paddle and state.ball_vel[0] < 0:
        ball_going_up = ball_vy < 0
        paddle_going_down = pad_vy > 0
        ball_going_down = ball_vy > 0
        paddle_going_up = pad_vy < 0

        opposite_direction = (ball_going_up and paddle_going_down) or (
            ball_going_down and paddle_going_up
        )

        if opposite_direction:
            velocity_strength = abs(pad_vy) / MAX_VELOCITY if MAX_VELOCITY > 0 else 0
            reward += 20 * velocity_strength

        elif ball_vy * pad_vy > 0:
            if MAX_VELOCITY > 0:
                ball_vy_normalized = ball_vy / MAX_VELOCITY
                pad_vy_normalized = pad_vy / MAX_VELOCITY
                velocity_match = 1 - abs(ball_vy_normalized - pad_vy_normalized)
                reward += velocity_match * 3

    reward -= paddle_speed_normalized * 2

    reward += state.score * 0.5

    return reward