import torch.nn as nn

from models.blocks import GlobalAvgPool2d


class _VanillaConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(_VanillaConvBlock, self).__init__()

        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self._block(x)


class VanillaCnn(nn.Module):
    def __init__(self, class_count=10, use_softmax=True):
        super(VanillaCnn, self).__init__()

        self._features = nn.Sequential(_VanillaConvBlock(in_channels=3, out_channels=8, kernel_size=3),
                                       _VanillaConvBlock(in_channels=8, out_channels=16, kernel_size=3),
                                       nn.MaxPool2d(kernel_size=2, stride=2),

                                       _VanillaConvBlock(in_channels=16, out_channels=32, kernel_size=3),
                                       _VanillaConvBlock(in_channels=32, out_channels=64, kernel_size=3),
                                       nn.MaxPool2d(kernel_size=2, stride=2),

                                       _VanillaConvBlock(in_channels=64, out_channels=128, kernel_size=3),
                                       _VanillaConvBlock(in_channels=128, out_channels=256, kernel_size=3),
                                       nn.MaxPool2d(kernel_size=2, stride=2))

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
