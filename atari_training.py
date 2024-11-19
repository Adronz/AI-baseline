import atari_player
import torch
import copy 
import numpy as np
from utils import plot_rewards, choose_epsilon
import torch.nn.functional as F
import imageio
import sys



def loss_fxn(q_net, q_target_net, experience_batch, device, gamma=0.99):
    '''creates a loss from the reward, q function, and the ideal q function,
    this is scaled by the time discount gamma '''
    obs, actions, rewards, next_obs, terminated= zip(*experience_batch)

    obs = torch.tensor(np.array(obs), dtype=torch.float32).squeeze(1).to(device)
    next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).squeeze(1).permute(0, 3, 1, 2).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    #* if terminated, target is only the reward
    terminated = torch.tensor(np.array(terminated), dtype= torch.float32).to(device)

    ####################* CALCULATE LOSS ########################
    with torch.no_grad():
        next_q_max, _ = torch.max(q_target_net(next_obs),dim=1) #* offline? selection

    td_target = rewards + gamma * next_q_max * (1 - terminated)

    #* the squeeze/unsqueeze business is so that the subtraction doesn't broadcast and make a bigger matrix
    #* but we also need the index in columns
    current_q_vals = q_net(obs)
    aligned_q_vals = torch.gather(current_q_vals, dim= 1, index=actions.unsqueeze(1).to(torch.int64)).squeeze(1)
    
    loss = torch.mean((td_target - aligned_q_vals)**2)

    return loss

def double_q_loss(q_net, q_target_net, experience_batch, device, gamma= 0.99):

    obs, actions, rewards, next_obs, terminated= zip(*experience_batch)

    #* divide by 255 to normalize
    obs = torch.tensor(np.array(obs), dtype=torch.float32).squeeze(1).to(device) / 255
    next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).squeeze(1).permute(0, 3, 1, 2).to(device) / 255

    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)

    #* if terminated, target is only the reward
    terminated = torch.tensor(np.array(terminated), dtype= torch.float32).to(device)

    # print(f'obs min: {obs.min()}, max: {obs.max()}, mean: {obs.mean()}, std: {obs.std()}')
    # print(f'next_obs min: {next_obs.min()}, max: {next_obs.max()}, mean: {next_obs.mean()}, std: {next_obs.std()}')
    

    with torch.no_grad():
        next_q_value = q_target_net(next_obs)

    #* dim=1 to avoid complete flattening, and keep dim so that gather works
    action_index = torch.argmax(q_net(next_obs), dim=1, keepdim= True)

    #* get the new target according to the DDQN paper
    #* this uses the online network to choose the action, but uses the q value from the offline network, effectively decoupling the two networks
    double_next_q_val = torch.gather(next_q_value, dim= 1, index=action_index).squeeze(1)
    double_td_target = rewards + gamma * double_next_q_val * (1 - terminated)

    current_q_vals = q_net(obs)
    aligned_q_vals = torch.gather(current_q_vals, dim= 1, index=actions.unsqueeze(1).to(torch.int64)).squeeze(1)

    double_q_loss = F.mse_loss(aligned_q_vals, double_td_target)

    return double_q_loss


def train(env, q_net, q_target_net, optimizer, replay_buffer, episodes, update_freq, batch_size, weight_pth, device):
    '''q_net is q_phi from the Berkeley RAIL lectures, and q_target_net is q_phi_prime'''
    action_counts = np.zeros(4)
    
    rewards_per_episode = []
    epsilon = 1
    frame_count = 0
    q_target_net.load_state_dict(torch.load(weight_pth, weights_only=True, map_location=device))


    for episode in range(episodes):
        #* update 'ideal' memory
        if frame_count % update_freq == 0:
                    torch.save(q_net.state_dict(), weight_pth)
                    q_target_net.load_state_dict(torch.load(weight_pth, weights_only=True, map_location=device))


        #* get the first frame 
        obs = torch.tensor(env.reset(), dtype=torch.float32).view(1, 4, 84, 84).to(device) / 255
        # obs = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        # obs = torch.tensor(obs, dtype=torch.float32).view(1, 4, 84, 84).to(device)


        #* track episode reward
        episode_reward = 0
        done = False
        #* train model for k steps, and then update bot
        while not done:
            q_net.train()
            optimizer.zero_grad()

            epsilon = choose_epsilon(epsilon)
            action = q_net.take_action(obs, epsilon, device) #* take action
            
            # action = np.int64(action.item()) #convert get the item int
            action = np.array([action.item()], dtype=np.int64)

            #* make sure its not favoring some action
            action_counts[action] += 1

            # print(f'my action shape is {action.shape}')
            # print(f'my action is {action}')

            next_obs, reward, done, _ = env.step(action)

            #*############### ADD EXPERIENCE TO BUFFER###############

            #* manage devices: env.step outputs np.array so we need everything to be on cpu 
            #* denormalize the observation and store as float16 to save memory (next_obs already [0,255])
            obs_f16 = (obs * 255).to(torch.float16).to('cpu')
            action = action.astype(np.int8)[0]
            reward = reward.astype(np.float16)[0]
            done_int = int(done[0])

            #* Add experience to replay buffer
            experience = (obs_f16, action, reward, next_obs, done_int)
            replay_buffer.push(experience)
        
            #*########################################################

            #* Normalize observation for next step
            next_obs = next_obs / 255
            next_obs = torch.tensor(next_obs, dtype=torch.float32).view(1, 4, 84, 84)

            #* fill half of memory for before training on a mini batch a minibatch
            if replay_buffer.__len__() > 50000:
                #sample from memory
                batch = replay_buffer.sample(batch_size)

                loss = double_q_loss(q_net, q_target_net, batch, device, gamma=0.99)
                loss.backward()
                optimizer.step()       

            #look at the next step in the game 
            obs = next_obs.to(device)
            done = done[0]

            #* update episode reward 
            episode_reward += reward.item()
            frame_count += 1
        
        rewards_per_episode.append(episode_reward)
        if episode % 1000 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}, Total Frames: {frame_count}, epsilon {epsilon}")  
            print(f"Action proportions: {action_counts}")
            plot_rewards(rewards_per_episode)
    
    print(f"Action proportions: {action_counts / episodes}")





def play(env, q_net, episodes, weight_pth, device, video_path='long_video.mp4', fps=30):
    '''Play the game and take a video'''
    action_counts = np.zeros(4)

    rewards_per_episode = []
    frame_count = 0
    q_net.load_state_dict(torch.load(weight_pth, map_location=device))


    for episode in range(episodes):
        #* get the first frame 
        obs = torch.tensor(env.reset(), dtype=torch.float32)
        obs = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)


        #* track episode reward
        episode_reward = 0
        done = False
        q_net.eval()
        all_frames = []


        #* train model for k steps, and then update bot
        while not done:

            with torch.no_grad():
                epsilon = 0
                action = q_net.take_action(obs, epsilon, device) #* take action
            print(action)
            action_counts[action] += 1
            
            # action = np.int64(action.item()) #convert get the item int
            action = np.array([action.item()], dtype=np.int64)
            


            next_obs, reward, done, _ = env.step(action)

            frame = env.render(mode='rgb_array')
            all_frames.append(frame)
            
            #* manage devices: env.step outputs np.array so we need everything to be on cpu 
            obs = obs.to('cpu')

            next_obs = torch.tensor(next_obs, dtype=torch.float32).permute(0, 3, 1, 2)

            #look at the next step in the game 
            obs = next_obs.to(device)
            done = done[0]

            #* update episode reward 
            episode_reward += reward.item()
            frame_count += 1
        
        rewards_per_episode.append(episode_reward)


    imageio.mimwrite(video_path, all_frames, fps=fps)
    print(f"Video saved to {video_path}")
    print(f"Action Counts: {action_counts}")


# def play(env, q_net, episodes, weight_pth, device, video_path='long_video.mp4', fps=30):
#     '''Play the game and record one long video'''
#     rewards_per_episode = []
#     frame_count = 0
#     q_net.load_state_dict(torch.load(weight_pth, map_location=device))
#     q_net.to(device)
#     q_net.eval()

#     # Initialize a list to store all frames
#     all_frames = []

#     for episode in range(episodes):
#         # Reset the environment and preprocess the initial observation
#         obs = env.reset()
#         obs = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

#         # Track episode reward
#         episode_reward = 0
#         done = False

#         while not done:
#             with torch.no_grad():
#                 # Select action with epsilon=0 for deterministic behavior
#                 action = q_net.take_action(obs, epsilon=0, device=device).item()

#             # Step the environment
#             next_obs, reward, done, _ = env.step(action)

#             # Render the current frame and append to the list
#             frame = env.render(mode='rgb_array')
#             all_frames.append(frame)

#             # Preprocess the next observation
#             next_obs = torch.tensor(next_obs, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)

#             # Update the observation and reward
#             obs = next_obs
#             episode_reward += reward
#             frame_count += 1

#         rewards_per_episode.append(episode_reward)
#         print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}")

#     # Save all frames as a single video
#     imageio.mimwrite(video_path, all_frames, fps=fps)
#     print(f"Video saved to {video_path}")

#     # Optionally, plot the rewards
#     plot_rewards(rewards_per_episode)