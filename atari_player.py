import torch
import torch.nn as nn
from collections import deque
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

class Replay_Buffer():
    def __init__(self,max_len):
        self.buffer = deque(maxlen=max_len)

    def push(self, new_experience):
        '''add a new experience to memory. The experience should be in the form:
        tuple(observation, action, reward, next observation)''' 
        self.buffer.append(new_experience)
    
    def sample(self, batch_size):
        '''Randomly sample from memory to avoid local overfitting'''
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)


class Atari_Agent(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.num_moves = num_moves
        self.network = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.ReLU(),
        # nn.Conv2d(32, 64, kernel_size=3, stride=1),
        # nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=32*9*9, out_features=256),
        nn.Linear(256, num_moves),
        )

    def forward(self, x):
        return self.network(x)
    
    def take_action(self, observation, epsilon, device=device):
        '''take the action that with the highest q_value. If the probability of exploring is
        less than epsilon, take a random action!'''
        p_exploration = random.random()
        if p_exploration > epsilon:
            q_value = self.forward(observation)
            print(q_value)
            action = torch.argmax(q_value)
            return action
        
        else:
            return random.randrange(0, self.num_moves)

        
