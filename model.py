import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torchvision.utils as vutils

class ResNetBlock(nn.Module):

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       nn.InstanceNorm2d(dim),
                       #nn.BatchNorm2d(dim),
                       nn.LeakyReLU(0.2),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       #nn.BatchNorm2d(dim)
                       nn.InstanceNorm2d(dim)
                       ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),

            nn.Conv2d(3, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            #ResNetBlock(256),
            #ResNetBlock(256),
            #ResNetBlock(256),
            #ResNetBlock(256),
            #ResNetBlock(256),
            #ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

        
        self.model.apply(self._init_weights)

    def forward(self, input):
        return self.model(input)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.nf = 64
    self.main = nn.Sequential(
        nn.Conv2d(3, self.nf, 4, 2, 1, bias = False),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Dropout(0.1),
        nn.Conv2d(self.nf, self.nf * 2, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 2),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Dropout(0.1),
        nn.Conv2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 4),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Dropout(0.1),
        nn.Conv2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 8),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Dropout(0.1),
        nn.Conv2d(self.nf * 8, self.nf * 16, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 16),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Dropout(0.1),
        nn.Conv2d(self.nf * 16, self.nf * 32, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 32),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Dropout(0.1),
        nn.Conv2d(self.nf * 32, 1, 4, 1, 0, bias = False),
        nn.Sigmoid()
    )
  def forward(self, input):
    output = self.main(input)
    return output.view(-1, 1).squeeze(1)

