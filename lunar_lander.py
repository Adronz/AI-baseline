import atari_player
import atari_training
import gymnasium as gym
import torch

env = gym.make("LunarLander-v3")
replay_buffer = atari_player.Replay_Buffer(10000)

model = atari_player.Atari_Agent(4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
atari_training.train(env, model, optimizer, replay_buffer, 1, 1, 1, device='cpu' )
