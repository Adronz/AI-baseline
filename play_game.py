import atari_player
import atari_training
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
import torch
import ale_py
from torchsummary import summary
import time
from utils import set_seed

start_time = time.time()
set_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

#* GPU Production Constants
# WEIGHT_PTH = "pong_model_gpu.pth"

#* CPU Testing Constants
EPISODES = 100
WEIGHT_PTH = "/dors/wankowicz_lab/adrian/kinase_colabfold/kinase_new_structs/breakout_model_gpu.pth"

# gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')

#make environment to DQN paper specs
env = AtariWrapper(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

#* video recording wrapper 
# env = VecVideoRecorder(
#     env,
#     video_folder='/dors/wankowicz_lab/adrian/temp/videos/',                # Directory to save videos
#     record_video_trigger=lambda episode_id: episode_id == 0,  # Record the first episode
#     video_length=10000,                     # Number of steps to record
#     name_prefix='breakout_test'                 # Prefix for video files
# )

action_space = env.action_space.n
print(action_space)

q_net = atari_player.Atari_Agent(action_space).to(device)

video_pth = '/dors/wankowicz_lab/adrian/temp/videos/breakout_game.mp4'

atari_training.play(env,
                    q_net,
                    EPISODES, 
                    WEIGHT_PTH,
                    device= device,
                    video_path= video_pth,
                    fps=30)

env.close()

elapsed_time = time.time() - start_time

elapsed_time = time.time() - start_time  # in seconds
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
hours = minutes // 60
minutes = minutes % 60

if hours > 0:
    print(f'Finished playing! It took {hours} hours, {minutes} minutes, and {seconds} seconds.')
else:
    print(f'Finished playing! It took {minutes} minutes and {seconds} seconds.')