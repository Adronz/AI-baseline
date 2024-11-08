import atari_player
import torch
import copy 

replay_buffer = atari_player.Replay_Buffer(1000)
#TODO create a loss function
def loss_fxn(q_target, experience_batch, gamma=0.01):
    '''creates a loss from the reward, q function, and the ideal q function,
    this is scaled by the time discount gamma '''
    obs, actions, rewards, next_obs, terminated= zip(*experience_batch)

    obs = torch.tensor(obs)
    next_obs = torch.tensor(next_obs, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    terminated = torch.tensor(terminated, dtype= torch.int) #if terminated, target is only the reward


    target_actions = q_target.take_action(obs,0.01).unsqueeze(1)
    q_values = torch.gather(target_actions, actions)


    #get target using the reward and the old network
    target = rewards + (1-terminated)*(gamma * q_target.take_action(next_obs, 0.01)) 

    loss = torch.mean((target - q_values)**2)

    return loss

#TODO create training loop function (by yourself dawg)
def train(env, model, optimizer, replay_buffer, episodes, T, batch_size, device):
    
    for episode in range(episodes):
        #update 'ideal' memory
        q_star = copy.deepcopy(model)
        #get the first frame 
        obs, info = env.reset()
        
        obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
        

        #train model for k steps, and then update bot
        for t in range(T):

            model.train()
            model.zero_grad()

            action = model.take_action(obs, 0.01)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = next_obs.permute(2, 0, 1)

            #add experience to replay buffer
            experience = tuple((obs, action, reward, next_obs, terminated))
            replay_buffer.push(experience)

            #make sure that there is enough memory for a minibatch
            if replay_buffer.len() >= batch_size:
                #sample from memory
                batch = replay_buffer.sample(batch_size)

                loss = loss_fxn(model, q_star, batch)
                loss.backward()
                optimizer.step()

                if t % 100 == 0:
                    q_star = model #! this might be scuffed

            #look at the next step in the game 
            obs = next_obs

            




