# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:14:10 2020

@author: Pankaj Mishra
"""

import torch.nn as nn


## Decoder ##

class decoder2(nn.Module):
    def __init__(self, in_channels):
        super(decoder2, self).__init__()
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=1),
            # In b, 8, 8, 8 >> out b, 16, 15, 15
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 9, stride=3, padding=1),  # out> b,32, 49, 49
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 7, stride=5, padding=1),  # out> b, 32, 245, 245
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 9, stride=2),  # out> b, 16, 497, 497
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 6, stride=1),  # out> b, 8, 502, 502
            nn.BatchNorm2d(8, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 11, stride=1),  # out> b, 3, 512, 512
            nn.Tanh()
        )

    def forward(self, x):
        recon = self.decoder2(x)
        return recon


class decoder28(nn.Module):
    def __init__(self, in_channels):
        super(decoder28, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=1),
            # In b, 8, 8, 8 >> out b, 16, 15, 15
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=2),  # out> b, 8, 29, 29
            nn.BatchNorm2d(8, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=1),  # out> b, 3, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        recon = self.decoder(x)
        return recon
