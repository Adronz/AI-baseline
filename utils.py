import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from datetime import datetime

def plot_rewards(episode_rewards, window_size=100):

    plt.figure()

    # Apply a moving average for smoothing
    smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')

    # Plot original and smoothed rewards
    plt.plot(episode_rewards, label='Original', alpha=0.3)
    plt.plot(smoothed_rewards, label='Smoothed', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title("Reward vs Episode")
    plt.grid()
    plt.legend()

    # Save with a timestamp
    save_path = f"/dors/wankowicz_lab/adrian/temp/reward_plot_gpu_test.png"
    plt.savefig(save_path)

    print(f"Plot saved to: {save_path}")

def choose_epsilon(epsilon, frame_count):
    if frame_count > 1000000:
        epsilon = 0.1
        return epsilon
    else:
        epsilon = epsilon - (1/1000000)
        return epsilon