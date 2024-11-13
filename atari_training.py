import atari_player
import torch
import copy 
import numpy as np

replay_buffer = atari_player.Replay_Buffer(1000)

def loss_fxn(q_net, q_target_net, experience_batch, gamma=0.01):
    '''creates a loss from the reward, q function, and the ideal q function,
    this is scaled by the time discount gamma '''
    obs, actions, rewards, next_obs, terminated= zip(*experience_batch)

    obs = torch.tensor(np.array(obs), dtype=torch.float32).squeeze(1)
    next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).squeeze(1)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).squeeze(1)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).squeeze(1)
    #* if terminated, target is only the reward
    terminated = torch.tensor(np.array(terminated), dtype= torch.int).squeeze(1) 

    ####################* CALCULATE LOSS ########################
    
    td_target = rewards + gamma * torch.max(q_target_net(next_obs))

    current_q_vals = q_net(obs)
    #* the squeeze/unsqueeze business is so that the subtraction doesn't broadcast and make a bigger matrix
    #* but we also need the index in columns
    aligned_q_vals = torch.gather(current_q_vals, dim= 1, index=actions.unsqueeze(1).to(torch.int64)).squeeze(1)
    
    loss = torch.mean((td_target - aligned_q_vals)**2)

    return loss


def train(env, q_net, q_target_net, optimizer, replay_buffer, episodes, update_freq, batch_size, device):
    '''q_net is q_phi from the Berkeley RAIL lectures, and q_target_net is q_phi_prime'''

    rewards_per_episode = []

    for episode in range(episodes):
        #* update 'ideal' memory
        q_target_net.load_state_dict(torch.load("breakout_model.pth"))
        #* get the first frame 
        obs = torch.tensor(env.reset(), dtype=torch.float32)
        obs = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2)

        #* track episode reward
        episode_reward = 0

        #* train model for k steps, and then update bot
        while not done:
            q_net.train()
            optimizer.zero_grad()

            #TODO make epsilon greedy decrease over time 
            action = q_net.take_action(obs, 0.99) #* take action ()
            
            # action = np.int64(action.item()) #convert get the item int
            action = np.array([action.item()], dtype=np.int64)
            # print(f'my action shape is {action.shape}')
            # print(f'my action is {action}')

            next_obs, reward, done, info = env.step(action)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).permute(0, 3, 1, 2)

            
            #add experience to replay buffer
            experience = (obs, action, reward, next_obs, done)
            replay_buffer.push(experience)

            #make sure that there is enough memory for a minibatch
            if replay_buffer.__len__() > batch_size:
                #sample from memory
                batch = replay_buffer.sample(batch_size)

                loss = loss_fxn(q_net, q_target_net, batch)
                loss.backward()
                optimizer.step()       

            #look at the next step in the game 
            obs = next_obs

            #* update episode reward 
            episode_reward += reward
            if done is True:
                rewards_per_episode.append(episode_reward)

        if episode % update_freq == 0:
                    torch.save(q_net.state_dict(), "breakout_model.pth")     
            




