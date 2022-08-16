import collections
import torch
from torch import nn


class SingleViewUSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([
            encoding_block(1, 32),
            encoding_block(32, 64)
        ])
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mid_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        )
        self.decoder = nn.ModuleList([
            decoding_block(64+128, 128),
            decoding_block(32+128, 128)
        ])
        self.final_block = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.shortcuts = collections.deque([])

    def forward(self, x):
        for en_layer in self.encoder:
            x = en_layer(x)
            x = self.pool2d(x)
            self.shortcuts.append(x)

        x = self.mid_block(x)

        for de_layer in self.decoder:
            shortcut = self.shortcuts.pop()
            x = torch.cat((shortcut, x), dim=1)
            x = de_layer(x)

        x = self.final_block(x)
        # Note that this final output is a Tensor of logits BEFORE it gets normalized
        # by the sigmoid function, it will get passed to torch.nn.BCEWithLogitsLoss
        # to get the binary cross entropy error

        return x


def encoding_block(in_features, out_features, activate='relu'):
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()]
    ])

    down_conv_block = nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_features),
        activations[activate]
    )
    return down_conv_block


def decoding_block(in_features, out_features, activate='relu'):
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()]
    ])

    up_conv_block = nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_features),
        activations[activate],
        nn.ConvTranspose2d(out_features, out_features, kernel_size=2, stride=2),
        nn.Dropout2d()
    )

    return up_conv_block


