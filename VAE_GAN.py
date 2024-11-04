print("SCRIPT HAS STARTED RUNNING")
print("this is the new one")

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import time 

start_time = time.time()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the transformation to convert the images to tensor
# Define the transformation to convert the images to tensor and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])
# Download and load the MNIST training dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Download and load the MNIST test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader can be used to create batches of tensors
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


def save_n_images(n, generator, file_dir, real_data, latent_channels):
    # Ensure the model uses its current device
    device = next(generator.parameters()).device  
    generator.eval()

    if n > 64:
        n = 64

    batch_size = 64

    # Create noise on the generator's device
    with torch.no_grad():
        logits, _, _ = generator(real_data)
            
    for i in range(n):
        file_name = f'{file_dir}/CIFAR10_real_long_{i}.png'
        img = logits[i].cpu()  # Move the image to CPU for saving
        plt.figure()
        save_image(img, file_name)

    for i in range(n):
        file_name = f'{file_dir}/CIFAR10_sample_long_{i}.png'
        img = generator.sample(batch_size, latent_channels, device)
        img = logits[i].cpu()  # Move the image to CPU for saving
        plt.figure()
        save_image(img, file_name)



    print('Saved Images!')


class VAE_Generator(nn.Module):
    def __init__(self, conv_dim, output_1, output_2, output_3, image_channels):
        super().__init__()
        #* important note: the input channels, and the input are different. the layer only cares about the # of channels, not batch size, HxW, etc
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, conv_dim, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),

            nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim*2),
            nn.ReLU(),

            nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim*4),
            nn.ReLU(),
        )

        self.mu_mlp = nn.Linear(2048, 2048) #! not sure about this layer dimension
        self.log_var_mlp = nn.Linear(2048, 2048)
        
        #* important note: the input channels, and the input are different. the layer only cares about the # of channels, not batch size, HxW, etc
        #*4x2x1 doubles the image size 
    
        self.generator = nn.Sequential(

        #layer 1
        nn.ConvTranspose2d(conv_dim * 4, output_1, kernel_size=4, stride=2, padding=1), 
        nn.BatchNorm2d(output_1),
        nn.ReLU(), # image size (output_1, 8, 8)

        nn.ConvTranspose2d(output_1, output_2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(output_2),
        nn.ReLU(), #image size(output_2, 16, 16)

        #*put this back if you are going up to 64x64
        # nn.ConvTranspose2d(output_2, output_3, kernel_size=4, stride=2, padding=1), #image size (output_3, 32, 32)
        # nn.BatchNorm2d(output_3), 
        # nn.ReLU(),

        nn.ConvTranspose2d(output_2, image_channels, kernel_size=4, stride=2, padding=1), #image size (image_channels, 28, 28)
        nn.Tanh() 
        )
        
    def forward(self, x):
        batch_size = x.shape[0]

        #get trainable layers
        mu, log_var = self.encode(x)

        #reparamiterization trick to allow effective sampling
        z = self.reparameterize(mu, log_var)

        #resize z to be the shape before it was flattened
        z = z.view(batch_size, 128, 4, 4) 

        generation = self.generator(z)

        return generation, mu, log_var
    
    #generator loss fxn
    def loss_function(self, D_g, output, target, mu, log_var, real_lth_layer, fake_lth_layer, epoch):

        # #* ENCODER LOSS
        lth_layer_sq_diff = (real_lth_layer - fake_lth_layer)**2
        log_prob_dx = -0.5 * lth_layer_sq_diff.sum(1) #The derivative of ln(x) = 1/x. 
        disc_l_loss = -log_prob_dx.mean()

        
        # recon_loss = F.binary_cross_entropy(output, target, reduction='sum')

        # KL_div = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))

        # print(f'kl div: {KL_div}')
        # beta = 4.0
        # kl_weight = min(1.0, epoch / 20.0)  # Gradually increase beta over the first 10 epochs
        # kl_effective = beta * KL_div
        # vae_loss = kl_effective

        #* GENERATOR LOSS

        #* TOTAL LOSS

        # total_loss = vae_loss + gan_loss + disc_l_loss


            # VAE Loss
        KL_div = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1), dim=0)
        kl_weight = min(1.0, epoch / 20.0)  # Gradually increase beta over the first 20 epochs
        vae_loss = kl_weight * KL_div

        # GAN Loss
        fake_target = torch.ones_like(D_g) * 0.95  # Label smoothing
        gan_loss = F.binary_cross_entropy(D_g, fake_target)

        recon_loss = F.mse_loss(output, target, reduction='mean')

        # Total Loss
        # total_loss = vae_loss + gan_loss + recon_loss 
        total_loss = vae_loss + gan_loss + disc_l_loss

        return total_loss
    
    def encode(self, input):
        # print(f"Input shape to encoder: {input.shape}")  # Debug statement

        encoding = self.encoder(input)

        # print(f'encoding shape is {encoding.shape}')

        #flatten encoding
        encoding = encoding.view(input.shape[0], -1)
        mu = self.mu_mlp(encoding)
        log_var = self.log_var_mlp(encoding)

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
    
    def sample(self,batch_size, latent_channels, device):

        sampled_z = torch.randn(batch_size, latent_channels, 4, 4, device=device)

        sampled_images = self.generator(sampled_z)

        return sampled_images


        

class DC_Discriminator(nn.Module):
    def __init__(self, conv_dim, image_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(image_channels, conv_dim, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_dim)
        
        self.conv2 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_dim * 2)
        
        self.conv3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_dim * 4)
        
        # self.conv4 = nn.Conv2d(conv_dim * 4, 1, kernel_size=3, stride=2, padding=1)
        
        self.fc = nn.Linear(conv_dim * 4 * 4 * 4, 1)

        # Sigmoid activation for final output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)

        # x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        # print(f'shape into fc of discriminator {x.shape}')

        lth_layer = x.clone()
        
        x = self.fc(x)
        x = self.sigmoid(x)

        return x, lth_layer

    def loss_function(self, D_g, D_x):
            epsilon = 1e-8

            #*CHANGING TO BCE LOSS
            fake_target = torch.zeros_like(D_g) + 0.05
            real_target = torch.ones_like(D_x) * 0.95

            fake_loss = F.binary_cross_entropy(D_g, fake_target) #we want D_g to go to zero
            real_loss = F.binary_cross_entropy(D_x, real_target) #we want D_x to go to one

            loss = fake_loss + real_loss

            return loss 



# Example usage
output_1 = 128
output_2 = 64
output_3 = 32
image_channels = 3  # For MNIST, which is grayscale
conv_dim = 32
latent_channels = 128

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


generator = VAE_Generator(conv_dim, output_1, output_2, output_3, image_channels).to(device)
discriminator = DC_Discriminator(conv_dim, image_channels).to(device)

# Apply the weights function to generator and discriminator
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

print('generator')
print(summary(generator, input_size=(3, 32, 32)))
print(discriminator)
print(summary(discriminator, input_size=(3, 32, 32)))
        


gen_lr = 2e-4
disc_lr = 2e-4

gen_optimizer = torch.optim.Adam(generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999))

# scheduler_G = StepLR(gen_optimizer, step_size=30, gamma=0.1)
# scheduler_D = StepLR(disc_optimizer, step_size=10, gamma=0.1)

gen_trn_loss = []
disc_trn_loss = []
gen_val_loss = []
disc_val_loss = [] 

EPOCHS = 50
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

            # Data handling
            real_data = train_data.view(batch_size, 3, 32, 32).to(device)
            fake_data, mu, log_var = generator(real_data)
            fake_data = fake_data.detach() #We dont want to keep the computation graph

            # Get discriminator probabilites per type of data
            real_prob, real_lth_layer = discriminator(real_data) 
            fake_prob, fake_lth_layer = discriminator(fake_data)

            # Train the discriminator 
            disc_loss = discriminator.loss_function(fake_prob, real_prob) 
            gen_loss = generator.loss_function(fake_prob, fake_data, real_data, mu, log_var, real_lth_layer, fake_lth_layer, epoch)

            disc_loss.backward()
            disc_optimizer.step()

            
            if batch_idx % k == 0:

                #regenerate samples
                fake_data, mu, log_var = generator(real_data)
                
                
                #regenerate probabilities
                real_prob, real_lth_layer = discriminator(real_data) 
                fake_prob, fake_lth_layer = discriminator(fake_data)

                #train generator 
                gen_loss = generator.loss_function(fake_prob, fake_data, real_data, mu, log_var, real_lth_layer, fake_lth_layer, epoch)

                gen_loss.backward()
                gen_optimizer.step()

            disc_trn_run += disc_loss.item()
            gen_trn_run += gen_loss.item()


    
    gen_trn_loss.append(gen_trn_run)
    disc_trn_loss.append(disc_trn_run)

    print(f"EPOCH {epoch + 1}/{EPOCHS}: Disc train loss: {disc_trn_run / len(train_loader):.4f}, Gen train loss: {gen_trn_run / len(train_loader):.4f} ")
print("training finished!")


data_iter = iter(test_loader)
images, labels = next(data_iter)
images = images.to(device)

base_path = '/dors/wankowicz_lab/adrian/kinase_colabfold/temp_outputs'
save_n_images(10, generator, base_path, images ,latent_channels)

elapsed_time = time.time() - start_time
print(f'Program took {elapsed_time / 60} minutes to run')
