import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class Net_3(nn.Module):
    def __init__(self, num_classes=10, num_groups=1):
        super(Net_3, self).__init__()

        self.in_channels = 8

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, 8)
        self.layer1 = self.make_layer(BasicBlock, 8, 2, stride=1, num_groups=num_groups)
        self.layer2 = self.make_layer(BasicBlock, 16, 2, stride=2, num_groups=num_groups)
        self.layer3 = self.make_layer(BasicBlock, 32, 2, stride=2, num_groups=num_groups)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride, num_groups):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, num_groups))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1, num_groups=num_groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
