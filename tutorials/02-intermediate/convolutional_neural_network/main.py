import argparse
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import wandb

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Settings / hyperparameters
# these defaults can be edited here or overwritten via command line
MODEL_NAME = ""
EPOCHS = 5
NUM_CLASSES = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.001
L1_SIZE = 16
L2_SIZE = 32

wandb.init(name=run_name, project="pytorch_tutorial")
#wandb.config.update({
#  "num_epochs" : num_epochs,
#  "batch_size" : batch_size,
#  "lr" : learning_rate,
#  "l1" : l1_size,
#  "l2" : l2_size
#})

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, wandb.config.l1_size, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(wandb.config.l1_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(wandb.config.l1_size, wandb.config.l2_size, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(wandb.config.l2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*wandb.config.l2_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)
model.cuda()
wandb.watch(model, log="all")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss" : loss}) 
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
	  
    acc = 100 * correct / total
    wandb.log({"acc" : acc})
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default=MODEL_NAME,
    help="Name of this model/run (model will be saved to this file)")
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=EPOCHS,
    help="Number of training epochs")
  parser.add_argument(
    "--l1_size",
    type=int,
    default=L1_SIZE,
    help="size of first conv layer")
  parser.add_argument(
    "--l2_size",
    type=int,
    default=L2_SIZE,
    help="size of second conv layer")
  args = parser.parse_args()
  train(args)
