import pickle
from unittest import case
from numpy import r_
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from modules.layers import *
import torch

class LeNet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = Conv2d(1, 6, kernel_size=5, stride=1)
        self.tanh1 = Tanh()
        self.avg_pool2d1 = AvgPool2d(kernel_size=(2,2))
        self.conv2 = Conv2d(6, 16, kernel_size=5, stride=1)
        self.tanh2 = Tanh()
        self.avg_pool2d2 = AvgPool2d(kernel_size=(2,2))
        self.conv3 = Conv2d(16, 120, kernel_size=5, stride=1)
        self.tanh3 = Tanh()
        self.fc1 = Linear(120, 84)
        self.tanh4 = Tanh()
        self.fc2 = Linear(84, 10)
        
    def CLRP(self, x, maxindex = [None]):
        if maxindex == [None]:
            maxindex = torch.argmax(x, dim=1)
        R = torch.ones(x.shape).cuda()
        R /= -self.num_classes
        for i in range(R.size(0)):
            R[i, maxindex[i]] = 1
        self.class_ind = R.detach().cpu().numpy()
        return R
    
    def LRP(self, x, maxindex = [None]):
        if maxindex == [None]:
            maxindex = torch.argmax(x, dim=1)
        R = torch.zeros(x.shape).cuda()
        for i in range(R.size(0)):
            R[i, maxindex[i]] = 1
        return R
    
    def forward(self, x, mode=-1):
        x = self.conv1(x)
        conv1 = self.tanh1(x)
        x = self.avg_pool2d1(conv1)
        x = self.conv2(x)
        conv2 = self.tanh2(x)
        x = self.avg_pool2d2(conv2)
        x = self.conv3(x)
        conv3 = self.tanh3(x)
        conv3 = conv3.view(-1, 120)
        x = self.fc1(conv3)
        fc1 = self.tanh4(x)
        x = self.fc2(fc1)
        if mode == 3:
            return conv1, x
        elif mode == 2:
            return conv2, x
        elif mode == 1:
            return conv3, x
        elif mode == 0:
            return fc1, x
        elif mode == -1:
            return x
        else:
            raise ValueError("check your mode value, it has a range from 0 to 3 integer")

    def relprop(self, R, alpha, flag="inter"):
        R = self.fc2.relprop(R, alpha)
        if flag == 'fc1': return R
        R = self.tanh4.relprop(R, alpha)
        R = self.fc1.relprop(R, alpha)
        R = R[:,:,None,None]
        if flag == 'conv3': return R
        R = self.tanh3.relprop(R, alpha)
        R = self.conv3.relprop(R, alpha)
        R = self.avg_pool2d2.relprop(R, alpha)
        if flag == 'conv2': return R
        R = self.tanh2.relprop(R, alpha)
        R = self.conv2.relprop(R, alpha)
        R = self.avg_pool2d1.relprop(R, alpha)
        if flag == 'conv1': return R
        R = self.tanh1.relprop(R, alpha)
        R = self.conv1.relprop(R, alpha)
        
        return R