import atari_player
import atari_training
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import torch
import ale_py
from torchsummary import summary
from time import time

start_time = time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

#Constants
EPISODES = 50
UPDATE_FREQ = 10
BATCH_SIZE = 32



# gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5")

#make environment to DQN paper specs
wrapped_env = AtariWrapper(env)
replay_buffer = atari_player.Replay_Buffer(500000)

env = DummyVecEnv([lambda: wrapped_env])
env = VecFrameStack(env, n_stack=4)

sample_action = env.action_space.sample()



q_net = atari_player.Atari_Agent(4).to(device)
q_target_net = atari_player.Atari_Agent(4).to(device)
torch.save(q_net.state_dict(), "breakout_model.pth") #* reset training

optimizer = torch.optim.Adam(q_net.parameters(), lr=0.001)
atari_training.train(env, q_net, q_target_net, optimizer, replay_buffer, EPISODES, UPDATE_FREQ, BATCH_SIZE, device=device)

elapsed_time = time() - start_time
print(f'Finished training! It took {elapsed_time / 60} minutes')