import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from datetime import datetime
import random
import torch

def plot_rewards(episode_rewards, window_size=100):

    plt.figure()

    # Apply a moving average for smoothing
    smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')

    # Plot original and smoothed rewards
    # plt.plot(episode_rewards, label='Original', alpha=0.3)
    plt.plot(smoothed_rewards, label='Smoothed', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title("Reward vs Episode")
    plt.grid()
    plt.legend()

    # Save with a timestamp
    save_path = f"/dors/wankowicz_lab/adrian/temp/sea_quest_reward_plot_gpu.png"
    plt.savefig(save_path)

def choose_epsilon(epsilon):
    if epsilon <= 0.05:
        epsilon = 0.05
        return epsilon
    else:
        epsilon = epsilon - (1/500000)
        return epsilon
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

