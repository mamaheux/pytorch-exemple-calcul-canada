import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import GlobalAvgPool2d

class _DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_channels, growth_rate):
        super(_DenseBlock, self).__init__()

        self._bns = nn.ModuleList()
        self._convs = nn.ModuleList()

        for i in range(nb_layers):
            kernel_size = 1 if i % 2 == 0 else 3
            padding = 0 if i % 2 == 0 else 1

            self._bns.append(nn.BatchNorm2d(in_channels + i * growth_rate))
            self._convs.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=kernel_size,
                                        padding=padding))

    def forward(self, x):
        output = x

        for i in range(len(self._convs)):
            input = output
            output = self._convs[i](F.relu(self._bns[i](input)))
            output = torch.cat([input, output], 1)

        return output


class DenseBlockCnn(nn.Module):
    def __init__(self, class_count=10, use_softmax=True):
        super(DenseBlockCnn, self).__init__()

        self._features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),

                                       _DenseBlock(nb_layers=2, in_channels=32, growth_rate=16),
                                       nn.MaxPool2d(kernel_size=2, stride=2),

                                       _DenseBlock(nb_layers=2, in_channels=64, growth_rate=32),
                                       nn.MaxPool2d(kernel_size=2, stride=2),

                                       _DenseBlock(nb_layers=2, in_channels=128, growth_rate=64),
                                       nn.MaxPool2d(kernel_size=2, stride=2)
                                       )
        classifier_layers = [
            GlobalAvgPool2d(),
            nn.Conv2d(256, class_count, kernel_size=1)
        ]
        if use_softmax:
            classifier_layers.append(nn.Softmax(dim=1))

        self._classifier = nn.Sequential(*classifier_layers)


    def forward(self, x):
        y = self._features(x)
        return self._classifier(y)[:, :, 0, 0]
