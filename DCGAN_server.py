print("SCRIPT HAS STARTED RUNNING")

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import csv
import math
import struct
import random
import matplotlib.pyplot as plt
import struct
from array import array
from os.path  import join
from torchvision.utils import save_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Define the transformation to convert the images to tensor
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Download and load the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader can be used to create batches of tensors
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

def save_n_images(n, generator, file_dir):
    generator.eval()
    if n > 64:
        n=64
        
    batch_size = 64

    noise = torch.randn(batch_size, input_dim, 1, 1, device=device)
    with torch.no_grad():
        logits = generator(noise)
    print(logits.shape)
    logits = logits.view(batch_size,28,28)

    for i in range(n):
        file_name=f'{file_dir}/MNIST_gen_{i}.png'
        img = logits[i].cpu()
        plt.figure()
        save_image(img, file_name)


class DC_Generator(nn.Module):
    def __init__(self, input_dim, output_1, output_2, output_3, image_channels):
        super().__init__()
        #* important note: the input channels, and the input are different. the layer only cares about the # of channels, not batch size, HxW, etc
        self.generator = nn.Sequential(

        #layer 1
        nn.ConvTranspose2d(input_dim, output_1, kernel_size=4, stride=1), 
        nn.BatchNorm2d(output_1),
        nn.ReLU(), # image size (output_1, 4, 4)

        nn.ConvTranspose2d(output_1, output_2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(output_2),
        nn.ReLU(), #image size(output_2, 7, 7)

        nn.ConvTranspose2d(output_2, output_3, kernel_size=4, stride=2, padding=1), #image size (output_3, 14, 14)
        nn.BatchNorm2d(output_3), 
        nn.ReLU(),

        nn.ConvTranspose2d(output_3, image_channels, kernel_size=4, stride=2, padding=1), #image size (image_channels, 28, 28)
        nn.Tanh() #4x2x1 doubles the image size 
        )
        
    def forward(self, x):
        output = self.generator(x)

        return output
    
    #generator loss fxn
    def loss_function(self, D_g):
        epsilon = 1e-8
                #minimize the correctness of the generator 
        # loss = -torch.mean(torch.log(D_g + epsilon))
    
        # loss = torch.mean(torch.log(1 - D_g + epsilon))

        #* CHANGE TO BCELOSS
        fake_target = torch.ones_like(D_g) * 0.95
        loss = F.binary_cross_entropy(D_g, fake_target)

        return loss

        

class DC_Discriminator(nn.Module):
    def __init__(self, output_1, output_2, output_3, image_channels):
        super().__init__()

        self.discriminator = nn.Sequential(
                nn.Conv2d(image_channels, output_3, kernel_size=4, stride=2, padding=1), 
                nn.BatchNorm2d(output_3), 
                nn.LeakyReLU(0.2),

                nn.Conv2d(output_3, output_2, kernel_size=4, stride=2, padding=1), 
                nn.BatchNorm2d(output_2), 
                nn.LeakyReLU(0.2),

                nn.Conv2d(output_2, output_1, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(output_1), 
                nn.LeakyReLU(0.2),

                nn.Conv2d(output_1, image_channels, kernel_size=3, stride=2, padding=1),
                nn.Sigmoid()
        )

    def forward(self, x):

        discrimination = self.discriminator(x)

        return discrimination

    def loss_function(self, D_g, D_x):
            epsilon = 1e-8

            #maximize the probability that the data is real
            # loss = -torch.mean(torch.log(D_x + epsilon) + torch.log(1 - D_g + epsilon)) #negative sign because we are doing gradient ASCENT

            #*CHANGING TO BCE LOSS
            fake_target = torch.zeros_like(D_g) + 0.05
            real_target = torch.ones_like(D_x) * 0.95

            fake_loss = F.binary_cross_entropy(D_g, fake_target) #we want D_g to go to zero
            real_loss = F.binary_cross_entropy(D_x, real_target) #we want D_x to go to one

            loss = fake_loss + real_loss

            return loss 


# Example usage
input_dim = 100  # Latent vector size
output_1 = 128
output_2 = 64
output_3 = 32
image_channels = 1  # For MNIST, which is grayscale

generator = DC_Generator(input_dim, output_1, output_2, output_3, image_channels).to(device)
discriminator = DC_Discriminator(output_1, output_2, output_3, image_channels).to(device)
        

from torch.optim.lr_scheduler import StepLR


disc_lr = 1e-3
gen_lr = 1e-4

gen_optimizer = torch.optim.Adam(generator.parameters(), lr= gen_lr)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr= disc_lr)


# scheduler_G = StepLR(gen_optimizer, step_size=30, gamma=0.1)
# scheduler_D = StepLR(disc_optimizer, step_size=10, gamma=0.1)

gen_trn_loss = []
disc_trn_loss = []
gen_val_loss = []
disc_val_loss = [] 

EPOCHS = 1
k = 1

for epoch in range(EPOCHS):
    gen_trn_run = 0.0
    disc_trn_run = 0.0
    gen_val_run = 0.0
    disc_val_run = 0.0

    for i in range(k):
        for batch_idx, (train_data, train_target) in enumerate(train_loader):
            discriminator.train()
            generator.train()
            gen_optimizer.zero_grad()  

            disc_optimizer.zero_grad()

            batch_size = train_data.shape[0]

            real_data = train_data.view(batch_size, 1, 28, 28).to(device)

            noise = torch.randn(batch_size, input_dim, 1, 1, device= device)

            fake_data = generator(noise).detach()

            real_prob = discriminator(real_data)
            fake_prob = discriminator(fake_data)

            disc_loss = discriminator.loss_function(fake_prob, real_prob) 
            gen_loss = generator.loss_function(fake_prob)


            disc_loss.backward()
            disc_optimizer.step()

            
            if batch_idx % k == 0:
                noise = torch.randn(batch_size, input_dim, 1, 1, device=device)
                fake_data = generator(noise)
                real_prob = discriminator(real_data)
                fake_prob = discriminator(fake_data)
                gen_loss = generator.loss_function(fake_prob)

                gen_loss.backward()
                gen_optimizer.step()

            disc_trn_run += disc_loss.item()
            gen_trn_run += gen_loss.item()


    
    gen_trn_loss.append(gen_trn_run)
    disc_trn_loss.append(disc_trn_run)

    print(f"EPOCH {epoch + 1}/{EPOCHS}: Disc train loss: {disc_trn_run / len(train_loader):.4f}, Gen train loss: {gen_trn_run / len(train_loader):.4f} ")
print("training finished!")



save_n_images(10, generator, '/dors/wankowicz_lab/adrian/kinase_colabfold/kinase_new_structs/temp_outputs')

