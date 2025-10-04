
from config import *
from nn import (train_step, create_observation, Actor, Critic, ReplayBuffer)
from pingpong import (process_physics, render, initialize_state, update_pid, State)

import cv2
import torch 

import numpy as np
from torch import nn 



def make_demo(actor):
    state = initialize_state(W_WIDTH, W_HEIGHT)
    for _ in range(4):
        done_demo = False
        while not done_demo:
            state, done_demo = process_physics(state, NN_DT, W_WIDTH, W_HEIGHT)
            obs = create_observation(state, "left")
            obs_tensor = torch.FloatTensor(obs.to_array()).unsqueeze(0)
            action = actor(obs_tensor).item()
            state.left_pad_vel[1] = action * PADDLE_VELOCITY_MAX
            state = update_pid(state)
            reward_ = reward_function(state)
            img = render(W_WIDTH, W_HEIGHT, state)

            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (W_WIDTH, 100), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            cv2.putText(img, f"DEMO MODE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, f"Action: {action:+.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.putText(img, f"Reward: {reward_}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Training", img)
            key = cv2.waitKey(12)
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()



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
            nn.Tanh,
            nn.BatchNorm1d,
        ])

        actor_main = torch.load("checkpoints/actor_15.875.pt" ,weights_only=False)
        actor_target = Actor()
        actor_target.load_state_dict(actor_main.state_dict())
        
        critic_main = torch.load("checkpoints/critic_15.875.pt" ,weights_only=False)
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
        noise_scale = 0.5

        old_score = 0
        restart_amount = 0

        best_mean = 0
        cur_sum = 0
        cur_mean = 0

        for step in range(1_000_00_000):
            
            obs = create_observation(state, "left")
            obs_tensor = torch.FloatTensor(obs.to_array()).unsqueeze(0)

            with torch.no_grad():
                action = actor_main(obs_tensor).item() + np.random.random(1)[0] * noise_scale

            state.left_pad_vel[1] = action * PADDLE_VELOCITY_MAX
            next_state, done = process_physics(state, NN_DT, W_WIDTH, W_HEIGHT)

            if not done:
                next_obs = create_observation(next_state, "left")
                reward = reward_function(state)
                episode_reward += reward
                buffer.add(obs.to_array(), action, reward, next_obs.to_array(), float(done))

                TRAIN_ONCE_IN = 32
                if restart_amount % TRAIN_ONCE_IN == 0 and restart_amount != 0: 
                    cur_mean = cur_sum / TRAIN_ONCE_IN
                    if cur_mean > best_mean:
                        best_mean = cur_mean
                        torch.save(actor_main, f"checkpoints/actor_{cur_mean:05.3f}.pt")
                        torch.save(critic_main, f"checkpoints/critic_{cur_mean:05.3f}.pt")
                        print(f'[---/---] Saving: {cur_mean:05.3f}')
                    cur_sum = 0  

                    losses = train_step(buffer, critic_target, critic_main, actor_main, actor_target, actor_optimizer, critic_optimizer, batch_size=64, )
                    if losses[0] is not None and step % 100 == 0:
                        print(f"Step {step}, Critic Loss: {losses[0]:.3f}, Actor Loss: {losses[1]:.3f}")

                next_state = update_pid(next_state)
                state = next_state

            if state.score >= 10:
                done = True

            if done:
                cur_sum += state.score
                print(f"[{restart_amount % DEMO_EACH:3d}/{DEMO_EACH}] reward: {reward:8.6f}, score: {state.score:2d}")
                #if restart_amount % DEMO_EACH == 0:
                #    make_demo(actor_main)

                step_in_observation = 0
                restart_amount += 1
                state = initialize_state(W_WIDTH, W_HEIGHT)
                episode_reward = 0
                old_score = 0

                actor_main.eval()
                actor_main.train()

            noise_scale *= 0.95
    except KeyboardInterrupt:
        pass

