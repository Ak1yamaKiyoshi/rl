
from config import *
from nn import (train_step, create_observation, Actor, Critic, ReplayBuffer)
from pingpong import (process_physics, render, initialize_state, update_pid, State)

import cv2
import torch 

import numpy as np
from torch import nn 


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
                    losses = train_step(buffer, critic_target, critic_main, actor_main, actor_target, actor_optimizer, critic_optimizer, batch_size=128, )
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