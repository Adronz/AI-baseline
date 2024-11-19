import atari_player
import atari_training
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import torch
import ale_py
from torchsummary import summary
import time
from utils import set_seed

start_time = time.time()
set_seed(12)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# #* GPU Production Constants
EPISODES = 40000
UPDATE_FREQ = 7000
BATCH_SIZE = 64
BUFFER_MEM = 1000000
# # WEIGHT_PTH = "/dors/wankowicz_lab/adrian/kinase_colabfold/kinase_new_structs/space_invaders_model_gpu.pth"
# # WEIGHT_PTH = "/dors/wankowicz_lab/adrian/kinase_colabfold/kinase_new_structs/sea_quest_model_gpu.pth"
WEIGHT_PTH = "/dors/wankowicz_lab/adrian/kinase_colabfold/kinase_new_structs/breakout_model_gpu.pth"


# #* CPU Testing Constants
# EPISODES = 100
# UPDATE_FREQ = 10
# BATCH_SIZE = 5
# BUFFER_MEM = 10000
# WEIGHT_PTH = "/dors/wankowicz_lab/adrian/kinase_colabfold/kinase_new_structs/breakout_model_cpu.pth"


# gym.register_envs(ale_py)
# env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
# env = gym.make('ALE/Seaquest-v5', render_mode='rgb_array')
env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')


#make environment to DQN paper specs
wrapped_env = AtariWrapper(env)
replay_buffer = atari_player.Replay_Buffer(BUFFER_MEM)

env = DummyVecEnv([lambda: wrapped_env])
env = VecFrameStack(env, n_stack=4)

action_space = env.action_space.n

# print(action_space)


q_net = atari_player.Atari_Agent(action_space).to(device)
q_target_net = atari_player.Atari_Agent(action_space).to(device)



# print(summary(q_net, (4, 84, 84)))

torch.save(q_net.state_dict(), WEIGHT_PTH ) #* reset training

optimizer = torch.optim.Adam(q_net.parameters(), lr=0.00025)
atari_training.train(env,
                    q_net,
                    q_target_net, 
                    optimizer, 
                    replay_buffer, 
                    EPISODES, 
                    UPDATE_FREQ, 
                    BATCH_SIZE, 
                    WEIGHT_PTH,
                    device=device)




elapsed_time = time.time() - start_time

elapsed_time = time.time() - start_time  # in seconds
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
hours = minutes // 60
minutes = minutes % 60

if hours > 0:
    print(f'Finished training! It took {hours} hours, {minutes} minutes, and {seconds} seconds.')
else:
    print(f'Finished training! It took {minutes} minutes and {seconds} seconds.')

