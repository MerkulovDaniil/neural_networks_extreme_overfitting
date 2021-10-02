import torch
import torch.nn as nn
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters 
input_size = 784
hidden_size = 512
num_classes = 10

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

global model
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

input_size_cifar = 32*32*3
global model_cifar
model_cifar = NeuralNet(input_size_cifar, hidden_size, num_classes).to(device)