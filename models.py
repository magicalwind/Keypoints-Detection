
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input 224, conv1_output (32,220,220), pool1_out (32,110,110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)
        # input 110, conv2_out (64,107,107), pool2_out (64,53,53)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)
        # input 53, conv3_out (128,51,51), pool3_out (128,25,25)
        self.conv3 = nn.Conv2d(64, 128,3)
        self.dropout1 = nn.Dropout2d(p =0.6)
        self.pool3 = nn.MaxPool2d(2,2)
        
        # flattened array 128*25*25 = 80000
        #self.dropout1 = nn.Dropout(p =0.2)
        self.fc1 = nn.Linear(128*25*25,300)
        self.fc1_bn = nn.BatchNorm1d(300)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(300,136)
        
        
        
        
        

        
    def forward(self, x):
        
        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool3(F.relu(self.dropout1(self.conv3(x))))
        x = x.view(x.size(0),-1)
        #x = self.dropout1(x)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
