import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from datetime import datetime


def plot_rewards(episode_rewards):
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title("Reward vs episode")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"/dors/wankowicz_lab/adrian/reward_plot_{timestamp}.png"
    plt.savefig(save_path)
    
    print(f"Plot saved to: {save_path}")