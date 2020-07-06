import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, (x.data.size()[2], x.data.size()[3]))
