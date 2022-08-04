import collections
import torch
from torch import nn


class UNet(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.encoder = nn.ModuleList([
            encoding_block(in_ch, 64),
            encoding_block(64, 128),
            encoding_block(128, 256)
        ])
        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.mid_block = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2, output_padding=0)
        )
        self.decoder = nn.ModuleList([
            decoding_block(256+512, 256),
            decoding_block(128+256, 128)
        ])

        self.final_block = final_block(64+128, 64)
        self.shortcuts = collections.deque([])

    def forward(self, x):
        for en_layer in self.encoder:
            x = en_layer(x)
            self.shortcuts.append(x)
            x = self.pool3d(x)

        x = self.mid_block(x)

        for de_layer in self.decoder:
            shortcut = self.shortcuts.pop()
            x = torch.cat((shortcut, x), dim=1)
            x = de_layer(x)

        shortcut = self.shortcuts.pop()
        x = torch.cat((shortcut, x), dim=1)
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
        nn.Conv3d(in_features, int(out_features/2), kernel_size=3, padding=1),
        nn.BatchNorm3d(int(out_features/2)),
        activations[activate],
        nn.Conv3d(int(out_features/2), out_features, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_features),
        activations[activate]
    )

    return down_conv_block


def decoding_block(in_features, out_features, activate='relu'):
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()]
    ])

    up_conv_block = nn.Sequential(
        nn.Conv3d(in_features, out_features, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_features),
        activations[activate],
        nn.Conv3d(out_features, out_features, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_features),
        activations[activate],
        nn.ConvTranspose3d(out_features, out_features, kernel_size=2, stride=2)
    )

    return up_conv_block


def final_block(in_features, out_features, activate='relu'):
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()]
    ])

    last_block = nn.Sequential(
        nn.Conv3d(in_features, out_features, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_features),
        activations[activate],
        nn.Conv3d(out_features, out_features, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_features),
        activations[activate],
        nn.Conv3d(out_features, 1, kernel_size=1)
    )

    return last_block

