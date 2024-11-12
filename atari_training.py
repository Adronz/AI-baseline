import atari_player
import torch
import copy 
import numpy as np

replay_buffer = atari_player.Replay_Buffer(1000)

def loss_fxn(q_target, experience_batch, gamma=0.01):
    '''creates a loss from the reward, q function, and the ideal q function,
    this is scaled by the time discount gamma '''
    obs, actions, rewards, next_obs, terminated= zip(*experience_batch)

    obs = torch.tensor(obs[0], dtype=torch.float32)
    next_obs = torch.tensor(next_obs[0], dtype=torch.float32)
    next_obs = next_obs.permute(0, 3, 1, 2) #put the channel in the correct place 
    actions = torch.tensor(actions[0], dtype=torch.float32)
    rewards = torch.tensor(rewards[0], dtype=torch.float32)
    terminated = torch.tensor(terminated[0], dtype= torch.int) #if terminated, target is only the reward

    #get the max_q and next max_q of the constant Q function 
    #using the lagging Q fxn means that the parameters can be accurately compared
    _, max_q = q_target.take_action(obs,0.01)
    _, next_max_q = q_target.take_action(next_obs, 0.01)
    #! gotta get the max

    #get target using the reward and the old network
    target = rewards + (1-terminated)*(gamma * next_max_q) 

    loss = torch.mean((target - max_q)**2)

    return loss


def train(env, model, optimizer, replay_buffer, episodes, T, batch_size, device):
    
    for episode in range(episodes):
        #update 'ideal' memory
        q_star = copy.deepcopy(model)
        #get the first frame 
        obs = torch.tensor(env.reset(), dtype=torch.float32)
        obs = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2)
        

        #train model for k steps, and then update bot
        for t in range(T):

            model.train()
            model.zero_grad()

            action, _ = model.take_action(obs, 0.01)
            
            # action = np.int64(action.item()) #convert get the item int
            action = np.array([action.item()], dtype=np.int64)
            # print(f'my action shape is {action.shape}')
            # print(f'my action is {action}')

            next_obs, rewards, dones, infos = env.step(action)
            
            #add experience to replay buffer
            experience = (obs, action, rewards, next_obs, dones)
            replay_buffer.push(experience)

            #make sure that there is enough memory for a minibatch
            if replay_buffer.__len__() >= batch_size:
                #sample from memory
                batch = replay_buffer.sample(batch_size)

                loss = loss_fxn(model, batch) #! model may need to be target model
                loss.backward()
                optimizer.step()

                if t % 100 == 0:
                    q_star = model #! this might be scuffed

            #look at the next step in the game 
            obs = next_obs

            




