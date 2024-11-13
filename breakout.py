import atari_player
import atari_training
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import torch
import ale_py
from torchsummary import summary

#Constants
EPISODES = 100
UPDATE_FREQ = 10
BATCH_SIZE = 32



# gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5")

#make environment to DQN paper specs
wrapped_env = AtariWrapper(env)
replay_buffer = atari_player.Replay_Buffer(10000)

env = DummyVecEnv([lambda: wrapped_env])
env = VecFrameStack(env, n_stack=4)

sample_action = env.action_space.sample()



q_net = atari_player.Atari_Agent(4)
q_target_net = atari_player.Atari_Agent(4)
torch.save(q_net.state_dict(), "breakout_model.pth") #* reset training
# # print(summary(model, input_size=(4, 84, 84)))

optimizer = torch.optim.Adam(q_net.parameters(), lr=0.001)
atari_training.train(env, q_net, q_target_net, optimizer, replay_buffer, EPISODES, UPDATE_FREQ, BATCH_SIZE, device='cpu')
print('Finished training!')