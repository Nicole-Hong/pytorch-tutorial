import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import wandb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
run_name = "gan example"
latent_size = 256
hidden_size_1 = 256
hidden_size_2 = 256
leaky_relu_const_1 = 0.2
leaky_relu_const_2 = 0.2

lr_d = 0.0002
lr_g = 0.0002
image_size = 784
num_epochs = 100
batch_size = 100
sample_dir = 'samples'
LOG_INTERVAL = 200

wandb.init(name=run_name, project="pytorch_gan_tutorial")

args = {
  "lr_d" : lr_d,
  "lr_g" : lr_g,
  "image_dim" : image_size,
  "epochs" : num_epochs,
  "batch_size" : batch_size,
  "sample_dir" : sample_dir
}

cfg = wandb.config
cfg.setdefaults(args)

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])
#mean=(0.5, 0.5, 0.5),   # 3 for RGB channels

# MNIST dataset
#mnist = torchvision.datasets.MNIST(root='../../data/',
# KMNIST dataset (Japanese handwritten characters)
mnist = torchvision.datasets.KMNIST(root='../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size_1),
    nn.LeakyReLU(leaky_relu_const_1),
    nn.Linear(hidden_size_1, hidden_size_2),
    nn.LeakyReLU(leaky_relu_const_2),
    nn.Linear(hidden_size_2, 1, bias=0),
    nn.Sigmoid())

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size_1),
    nn.ReLU(),
    nn.Linear(hidden_size_1, hidden_size_2),
    nn.ReLU(),
    nn.Linear(hidden_size_2, image_size),
    nn.Tanh())

# experiment with labeling gradient plots
wandb.watch((D, G), log="all", labels=["discriminator", "generator"])
#wandb.watch((D, G), log="all")

# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=args['lr_d'])
g_optimizer = torch.optim.Adam(G.parameters(), lr=args['lr_g'])

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % LOG_INTERVAL == 0:
            wandb.log({
              "d_loss_real" : d_loss_real.item(), 
              "d_loss_fake" : d_loss_fake.item(),
              "d_loss_total" : d_loss.item(),
              "g_loss" : g_loss.item(),
              "D(x)" : real_score.mean().item(),
              "D(G(z))" : fake_score.mean().item()
            })
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))
    wandb.log({"examples" : [wandb.Image(f) for f in fake_images[:10]]})

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
