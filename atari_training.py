import atari_player
import torch
import copy 
import numpy as np
from utils import plot_rewards, choose_epsilon

replay_buffer = atari_player.Replay_Buffer(1000)

def loss_fxn(q_net, q_target_net, experience_batch, device, gamma=0.9):
    '''creates a loss from the reward, q function, and the ideal q function,
    this is scaled by the time discount gamma '''
    obs, actions, rewards, next_obs, terminated= zip(*experience_batch)

    obs = torch.tensor(np.array(obs), dtype=torch.float32).squeeze(1).to(device)
    next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).squeeze(1).permute(0, 3, 1, 2).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.int64).squeeze(1).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).squeeze(1).to(device)
    #* if terminated, target is only the reward
    terminated = torch.tensor(np.array(terminated), dtype= torch.int).squeeze(1).to(device)

    ####################* CALCULATE LOSS ########################
    with torch.no_grad():
        td_target = rewards + gamma * torch.max(q_target_net(next_obs)) * (1 - terminated)

    current_q_vals = q_net(obs)

    #* the squeeze/unsqueeze business is so that the subtraction doesn't broadcast and make a bigger matrix
    #* but we also need the index in columns
    aligned_q_vals = torch.gather(current_q_vals, dim= 1, index=actions.unsqueeze(1).to(torch.int64)).squeeze(1)
    
    loss = torch.mean((td_target - aligned_q_vals)**2)

    return loss


def train(env, q_net, q_target_net, optimizer, replay_buffer, episodes, update_freq, batch_size, weight_pth, device):
    '''q_net is q_phi from the Berkeley RAIL lectures, and q_target_net is q_phi_prime'''

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
        obs = torch.tensor(env.reset(), dtype=torch.float32)
        obs = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

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
            # print(f'my action shape is {action.shape}')
            # print(f'my action is {action}')

            next_obs, reward, done, _ = env.step(action)
            
            #* manage devices: env.step outputs np.array so we need everything to be on cpu 
            obs = obs.to('cpu')
            
            #add experience to replay buffer
            experience = (obs, action, reward, next_obs, done)
            replay_buffer.push(experience)

            next_obs = torch.tensor(next_obs, dtype=torch.float32).permute(0, 3, 1, 2)


            #* fill half of memory for before training on a mini batch a minibatch
            if replay_buffer.__len__() > 50000:
                #sample from memory
                batch = replay_buffer.sample(batch_size)

                loss = loss_fxn(q_net, q_target_net, batch, device, gamma=0.9)
                loss.backward()
                optimizer.step()       

            #look at the next step in the game 
            obs = next_obs.to(device)
            done = done[0]

            #* update episode reward 
            episode_reward += reward.item()
            frame_count += 1
        
        rewards_per_episode.append(episode_reward)
        if episode % 500 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}, Total Frames: {frame_count}, epsilon {epsilon}")  

    plot_rewards(rewards_per_episode)



