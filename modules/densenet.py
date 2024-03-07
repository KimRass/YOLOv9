# References:
    # https://deep-learning-study.tistory.com/545

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        inner_channels = 4 * growth_rate

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inner_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, growth_rate, 3, 1, 1, bias=False)
        ) # "$H_{l}$"

    def forward(self, x):
        return torch.cat([x, self.layers(x)], dim=1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)
