import atari_player
import atari_training
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import torch
import ale_py
from torchsummary import summary


# gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5")

#make environment to DQN paper specs
wrapped_env = AtariWrapper(env)
replay_buffer = atari_player.Replay_Buffer(10000)

env = DummyVecEnv([lambda: wrapped_env])
env = VecFrameStack(env, n_stack=4)

sample_action = env.action_space.sample()



model = atari_player.Atari_Agent(4)
# # print(summary(model, input_size=(4, 84, 84)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
atari_training.train(env, model, optimizer, replay_buffer, 1, 1, 1, device='cpu')