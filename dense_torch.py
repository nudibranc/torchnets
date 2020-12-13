import math
import torch 
import torch.nn as nn
import torch.nn.functional as functional

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate = 0.0):
        super(BasicBlock, self).__init__()
        self.bn =  nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)    
        self.conv1 = nn.Conv2d(in_planes,out_planes, 3,stride=1,padding=1,bias=False)
        self.dropRate = dropRate
    def forward(self,x):
        out = self.conv1(self.relu(self.bn(x)))
        if dropRate > 0: out = F.dropout(out, p=self.dropRate, training = self.training)
        return torch.cat([x,out], 1)