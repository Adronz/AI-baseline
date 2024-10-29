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
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Download and load the MNIST test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

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


class VAE(nn.Module):
    def __init__(self, conv_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
                nn.Conv2d(3, conv_dim, kernel_size=3, stride=2, padding=1), # output size = conv_dim, 
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(),

                nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(conv_dim*2),
                nn.ReLU(),

                nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(conv_dim*4),
                nn.ReLU(),

                nn.Flatten(), #make it so this can be split into mu and log_var

        )

        self.mu_mlp = nn.Linear(2048, 2048)
        self.var_mlp = nn.Linear(2048, 2048)

    def forward(self, x):
        #batch size, h * w
        encoding = self.encode(x)
        mu, log_var = encoding
        # print(f'decoded shape: {decoded.shape}')
        z = self.reparameterize(mu, log_var)
        
        return mu, log_var, z

    def encode(self, input):
        encoding = self.encoder(input)
        mu = self.mu_mlp(encoding)
        log_var = self.var_mlp(encoding)

        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        # Reparameterization trick:
        # let z ~= q_psi(z|x)
        # reparamterize z = g_psi(x, epsilon)
        # z = mu_i + sig_i * epsilon


        sig = torch.exp(log_var * 0.5)
        # print(sig)
        
        epsilon = torch.randn_like(sig)  # Random noise from a standard normal distribution.
        z = mu + torch.mul(sig, epsilon)

        return z
    
    def loss_function(self, output, target, mu, log_var, epoch):
        #here is the bread and butter
        #were looking to optimize mu and sigma
        
        recon_loss = F.binary_cross_entropy(output, target, reduction='sum')

        KL_div = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))

        # print(f'kl div: {KL_div}')
        beta = 4.0
        kl_weight = min(1.0, epoch / 20.0)  # Gradually increase beta over the first 10 epochs
        kl_effective = beta * KL_div
        loss = kl_effective + recon_loss
        

        return loss, kl_effective, recon_loss

    def sample(self):

        sampled_z = torch.randn(self.latent_dim)

        sampled_images = self.decoder(sampled_z)

        return sampled_images


class VAE_Generator(nn.Module):
    def __init__(self, conv_dim, input_dim, output_1, output_2, output_3, image_channels):
        super().__init__()
        #* important note: the input channels, and the input are different. the layer only cares about the # of channels, not batch size, HxW, etc
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, conv_dim, kernel_size=3, stride=2, padding=1), # output size = conv_dim, 
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),

            nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim*2),
            nn.ReLU(),

            nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim*4),
            nn.ReLU(),

            nn.Flatten(), #make it so this can be split into mu and log_var

        )

        self.mu_mlp = nn.Linear(2048, 2048) #! not sure about this layer dimension
        self.var_mlp = nn.Linear(2048, 2048)
        
        
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
        encoding = self.encode(x)
        mu, log_var = encoding
        # print(f'decoded shape: {decoded.shape}')
        z = self.reparameterize(mu, log_var)

        generation = self.generator(z)

        return generation
    
    #generator loss fxn
    def loss_function(self, D_g, output, target, mu, log_var, epoch):

        #* ENCODER LOSS
        #! replace recon loss with gaussian similarity 
        disc_l_loss = 0
        recon_loss = F.binary_cross_entropy(output, target, reduction='sum')

        KL_div = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))

        # print(f'kl div: {KL_div}')
        beta = 4.0
        kl_weight = min(1.0, epoch / 20.0)  # Gradually increase beta over the first 10 epochs
        kl_effective = beta * KL_div
        vae_loss = kl_effective + recon_loss

        #* GENERATOR LOSS
        epsilon = 1e-8
                #minimize the correctness of the generator 
        # loss = -torch.mean(torch.log(D_g + epsilon))
    
        # loss = torch.mean(torch.log(1 - D_g + epsilon))

        fake_target = torch.ones_like(D_g) * 0.95
        gan_loss = F.binary_cross_entropy(D_g, fake_target)
        
        #* TOTAL LOSS

        total_loss = vae_loss + gan_loss

        return total_loss
    
    def encode(self, input):
        encoding = self.encoder(input)
        mu = self.mu_mlp(encoding)
        log_var = self.var_mlp(encoding)

        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        # Reparameterization trick:
        # let z ~= q_psi(z|x)
        # reparamterize z = g_psi(x, epsilon)
        # z = mu_i + sig_i * epsilon


        sig = torch.exp(log_var * 0.5)
        # print(sig)
        
        epsilon = torch.randn_like(sig)  # Random noise from a standard normal distribution.
        z = mu + torch.mul(sig, epsilon)

        return z
    
    def sample(self):

        sampled_z = torch.randn(self.latent_dim)

        sampled_images = self.decoder(sampled_z)

        return sampled_images


        

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

                nn.Conv2d(output_1, 1, kernel_size=3, stride=2, padding=1),
                nn.Flatten(),

                nn.Linear(16, 1),
                nn.Sigmoid()
        )

    def forward(self, x):

        discrimination = self.discriminator(x)
        print(discrimination.shape)



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

generator = VAE_Generator(32, input_dim, output_1, output_2, output_3, image_channels).to(device)
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

