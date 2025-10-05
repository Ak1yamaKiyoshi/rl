import cv2
import numpy as np
from dataclasses import dataclass


from config import *
from pid import PIDController
from typing import Tuple


@dataclass
class State:
    ball_pos: np.ndarray  # center
    ball_vel: np.ndarray
    left_pad_pos: np.ndarray  # center
    left_pad_vel: np.ndarray
    right_pad_pos: np.ndarray  # center
    right_pad_vel: np.ndarray
    score: int
    pad_size: np.ndarray
    pid_controller: PIDController


def initialize_state(width, height) -> State:
    pads_y = height // 2
    left_pad_x = 10
    right_pad_x = width - 10
    ball_pos = np.array([width // 2, height // 2], dtype=np.float32)

    speed = 350
    angle = (np.random.random() - 0.5) * (np.pi / 2)
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
        pid_controller=PIDController(0.4, 0.05, 0.0),
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

    left_minmax_y = np.array(
        [
            new_left_pad_position[1] - state.pad_size[1] // 2,
            new_left_pad_position[1] + state.pad_size[1] // 2,
        ]
    )
    right_minmax_y = np.array(
        [
            new_right_pad_position[1] - state.pad_size[1] // 2,
            new_right_pad_position[1] + state.pad_size[1] // 2,
        ]
    )

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
        crossed_left = (old_ball_x >= x_boundary_left) and (
            new_ball_pos[0] <= x_boundary_left
        )

        if crossed_left and np.abs(new_ball_pos[0] - old_ball_x) > 1e-6:
            t_left = (x_boundary_left - old_ball_x) / (new_ball_pos[0] - old_ball_x)
            t_left = float(np.clip(t_left, 0, 1))

            intersection_y_left = old_ball_y + t_left * (new_ball_pos[1] - old_ball_y)

            hit_left_paddle = (intersection_y_left >= left_minmax_y[0]) and (
                intersection_y_left <= left_minmax_y[1]
            )

            if hit_left_paddle:
                ball_reflected = True

                hit_position_left = float(
                    (intersection_y_left - new_left_pad_position[1])
                    / (state.pad_size[1] / 2)
                )
                hit_position_left = np.clip(hit_position_left, -1.0, 1.0)

                new_vx = -new_ball_velocity[0]

                angle_deflection = hit_position_left * (np.pi / 6)

                paddle_influence = float(new_left_pad_vel[1]) * 0.3

                new_vy = (
                    new_ball_velocity[1]
                    + np.tan(angle_deflection) * np.abs(new_vx)
                    + paddle_influence
                )

                new_ball_velocity = np.array([new_vx, new_vy], dtype=np.float32)

                current_speed = np.linalg.norm(new_ball_velocity)
                if current_speed > 0:
                    new_ball_velocity = new_ball_velocity * (
                        original_speed / current_speed
                    )

                remaining_time = 1.0 - t_left
                new_ball_pos = (
                    np.array([x_boundary_left, intersection_y_left], dtype=np.float32)
                    + new_ball_velocity * dt * remaining_time
                )

                new_score = new_score + 1

    if new_ball_velocity[0] > 0 and not ball_reflected:
        crossed_right = (old_ball_x <= x_boundary_right) and (
            new_ball_pos[0] >= x_boundary_right
        )

        if crossed_right and np.abs(new_ball_pos[0] - old_ball_x) > 1e-6:
            t_right = (x_boundary_right - old_ball_x) / (new_ball_pos[0] - old_ball_x)
            t_right = float(np.clip(t_right, 0, 1))

            intersection_y_right = old_ball_y + t_right * (new_ball_pos[1] - old_ball_y)

            hit_right_paddle = (intersection_y_right >= right_minmax_y[0]) and (
                intersection_y_right <= right_minmax_y[1]
            )

            if hit_right_paddle:
                ball_reflected = True

                hit_position_right = float(
                    (intersection_y_right - new_right_pad_position[1])
                    / (state.pad_size[1] / 2)
                )
                hit_position_right = np.clip(hit_position_right, -1.0, 1.0)

                new_vx = -new_ball_velocity[0]

                angle_deflection = hit_position_right * (np.pi / 6)

                paddle_influence = float(new_right_pad_vel[1]) * 0.3

                new_vy = (
                    new_ball_velocity[1]
                    + np.tan(angle_deflection) * np.abs(new_vx)
                    + paddle_influence
                )

                new_ball_velocity = np.array([new_vx, new_vy], dtype=np.float32)

                current_speed = np.linalg.norm(new_ball_velocity)
                if current_speed > 0:
                    new_ball_velocity = new_ball_velocity * (
                        original_speed / current_speed
                    )

                remaining_time = 1.0 - t_right
                new_ball_pos = (
                    np.array([x_boundary_right, intersection_y_right], dtype=np.float32)
                    + new_ball_velocity * dt * remaining_time
                )

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


def render(height, width, state: State):
    canvas = np.zeros([width, height, 3])
    cv2.circle(canvas, list(map(int, state.ball_pos.tolist())), 5, (255, 255, 255))
    cv2.putText(
        canvas, str(state.score), [width // 2, height // 3], 0, 1, (255, 255, 255)
    )

    for pad in [state.left_pad_pos, state.right_pad_pos]:
        x, y = pad
        pt1 = int(x - state.pad_size[0] // 2), int(y - state.pad_size[1] // 2)
        pt2 = int(x + state.pad_size[0] // 2), int(y + state.pad_size[1] // 2)
        cv2.rectangle(canvas, pt1, pt2, (255, 255, 255, -1))
    return canvas


def update_pid(state: State, dt=NN_DT):
    measurement = -(state.ball_pos[1] - state.right_pad_pos[1])
    ctl = state.pid_controller.update(measurement, dt)
    state.right_pad_vel[1] = np.clip(ctl * 100, -1000, 1000)
    return state
