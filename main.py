from config import *
from nn import (train_step, create_observation, Actor, Critic, ReplayBuffer, 
                Observation, ObservationBuffer, add_noise)
from pingpong import (process_physics, render, initialize_state, update_pid, State)

import cv2
import torch 
import numpy as np
from torch import nn 


def make_demo(actor, noise_scale, device):
    state = initialize_state(W_WIDTH, W_HEIGHT)
    for _ in range(4):
        done_demo = False
        buffer = ObservationBuffer([])
        buffer.initialize()
        while not done_demo:
            state, done_demo = process_physics(state, NN_DT, W_WIDTH, W_HEIGHT)
            buffer.add(create_observation(state, "left"))
            obs_tensor = buffer.to_tensor().to(device)
            
            with torch.no_grad():
                action = actor(obs_tensor).item()
            
            state.left_pad_vel[1] = action * PADDLE_VELOCITY_MAX
            state = update_pid(state)
            reward_ = reward_function(state)
            
            img = render(W_WIDTH, W_HEIGHT, state)
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (W_WIDTH, 120), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            
            cv2.putText(img, f"DEMO MODE", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, f"Action: {action:+.3f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.putText(img, f"Reward: {reward_:.2f}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"Noise: {noise_scale:.4f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Training", img)
            key = cv2.waitKey(12)
            
            if key == ord('q') or state.score > 15:
                cv2.destroyAllWindows()
                return
    
    cv2.destroyAllWindows()


def evaluate_actor(actor, num_games=100, device='cpu'):
    total_score = 0
    
    for game in range(num_games):
        state = initialize_state(W_WIDTH, W_HEIGHT)
        buffer = ObservationBuffer([])
        buffer.initialize()
        done = False
        
        while not done:
            state, done = process_physics(state, NN_DT, W_WIDTH, W_HEIGHT)
            buffer.add(create_observation(state, "left"))
            obs_tensor = buffer.to_tensor().to(device)  
            
            with torch.no_grad():
                action = actor(obs_tensor).item()
            
            action = np.clip(action, -1.0, 1.0)
            state.left_pad_vel[1] = action * PADDLE_VELOCITY_MAX
            state = update_pid(state)
            
            if state.score >= 10:
                done = True
        
        total_score += state.score
    
    return total_score / num_games
if __name__ == "__main__":     
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        torch.set_num_threads(12)
        torch.set_flush_denormal(True)

        torch.serialization.add_safe_globals([
            Actor, 
            Critic,
            nn.Sequential,
            nn.Linear,
            nn.ReLU,
            nn.Tanh,
            nn.BatchNorm1d,
        ])

        actor_main = Actor().to(device)
        actor_target = Actor().to(device)
        actor_target.load_state_dict(actor_main.state_dict())

        critic_main = Critic().to(device)
        critic_target = Critic().to(device)
        critic_target.load_state_dict(critic_main.state_dict())

        actor_optimizer = torch.optim.Adam(actor_main.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(critic_main.parameters(), lr=1e-4)

        actor_target.eval()
        critic_target.eval()

        for param in actor_target.parameters():
            param.requires_grad = False
        for param in critic_target.parameters():
            param.requires_grad = False

        buffer = ReplayBuffer(capacity=1_000_000, device=device)
        state = initialize_state(W_WIDTH, W_HEIGHT)

        noise_scale = 0.5
        noise_decay = 0.9995
        min_noise = 0.01

        episode_count = 0
        best_eval_score = 0

        observation_buffer = ObservationBuffer([])
        observation_buffer.initialize()

        print(f"Initial noise scale: {noise_scale}")

        for step in range(1_000_000_000):
            observation_buffer.add(create_observation(state, 'left'))
            obs_tensor = observation_buffer.to_tensor().to(device)

            with torch.no_grad():
                action = actor_main(obs_tensor).item()
            
            action = add_noise(action, noise_scale)
            action = np.clip(action, -1.0, 1.0)

            state.left_pad_vel[1] = action * PADDLE_VELOCITY_MAX
            next_state, done = process_physics(state, NN_DT, W_WIDTH, W_HEIGHT)

            if not done:
                next_obs_buff = observation_buffer.copy()
                next_obs_buff.add(create_observation(next_state, "left"))
                reward = reward_function(next_state)
                
                buffer.add(
                    observation_buffer.to_numpy().reshape(-1), 
                    action, 
                    reward, 
                    next_obs_buff.to_numpy(), 
                    float(done)
                )

                if step % TRAIN_EVERY == 0:
                    losses = train_step(
                        buffer, critic_target, critic_main, 
                        actor_main, actor_target, 
                        actor_optimizer, critic_optimizer, 
                        batch_size=TRAIN_BATCH_SIZE,
                        device=device 
                    )
                    
                    if losses[0] is not None and step % 100 == 0:
                        current_lr = actor_optimizer.param_groups[0]['lr']

                        print(f"Step {step:7d} | C-Loss: {losses[0]:6.3f} | "
                            f"A-Loss: {losses[1]:7.3f} | Lr: {current_lr} | Noise: {noise_scale:.4f}")

                next_state = update_pid(next_state)
                state = next_state

            if state.score >= 10:
                done = True

            if done:
                observation_buffer = ObservationBuffer([])
                observation_buffer.initialize()
                
                episode_count += 1
                print(f"[Episode {episode_count:4d}] Score: {state.score:2d} | "
                    f"Noise: {noise_scale:.4f} Reward: {reward}")

                if episode_count % DEMO_EACH == 0:
                    make_demo(actor_main, noise_scale, device=device)  
                
                if episode_count % EVAL_EACH == 0:
                    actor_main.eval()
                    eval_score = evaluate_actor(actor_main, num_games=100)
                    actor_main.train()
                    
                    print(f"Evaluation score: {eval_score:.2f}")
                    
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        torch.save(actor_main, f"checkpoints/actor_{eval_score:.2f}.pt")
                        torch.save(critic_main, f"checkpoints/critic_{eval_score:.2f}.pt")
                        print(f" Saved checkpoint: {eval_score:.2f} ***\n")
                    else:
                        print(f"No improvement (best: {best_eval_score:.2f})\n")
                
                state = initialize_state(W_WIDTH, W_HEIGHT)
                
                noise_scale = max(min_noise, noise_scale * noise_decay)


                
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Final noise scale: {noise_scale:.4f}")
        print(f"Best evaluation score: {best_eval_score:.2f}")