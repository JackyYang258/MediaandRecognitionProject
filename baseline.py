import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        channels = 3
        self.conv1 = nn.Conv2d(1*channels, 6*channels, 5)
        self.bn1 = nn.BatchNorm2d(6*channels)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6*channels, 16*channels, 5)
        self.bn2 = nn.BatchNorm2d(16*channels)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*5*5*channels, 120*channels)
        self.bn3 = nn.BatchNorm1d(120*channels)
        self.fc2 = nn.Linear(120*channels, 84*channels)
        self.bn4 = nn.BatchNorm1d(84*channels)
        self.fc3 = nn.Linear(84*channels, 10)
        
        self.dropout = nn.Dropout(0.3)
        # self.conv1 = nn.Conv2d(1*channels, 6, 5)
        # self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(16*5*5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
    def forward(self,x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.pool1(output)
        
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.pool2(output)
        
        output = output.view(output.shape[0], -1)
        output = self.fc1(output)
        output = self.bn3(output)
        output = self.relu(output)
        
        output = self.fc2(output)
        output = self.bn4(output)
        output = self.relu(output)
        output = self.fc3(output)
        # output = self.conv1(x)
        # output = self.tanh(output)
        # output = self.pool1(output)
        # output = self.conv2(output)
        # output = self.tanh(output)
        # output = self.pool2(output)
        # output = output.view(output.shape[0], -1)
        # output = self.fc1(output)
        # output = self.tanh(output)
        # output = self.fc2(output)
        # output = self.tanh(output)
        # output = self.fc3(output)
        
        return output