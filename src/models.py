import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# ---------------------------------------------------------
#           S8 Models
# ---------------------------------------------------------
class BasicBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class Net_11(nn.Module):
    def __init__(self, num_classes=10):
        super(Net_11, self).__init__()

        self.in_channels = 8

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.layer1 = self.make_layer(BasicBlock1, 8, 2, stride=1)
        self.drop = nn.Dropout(0.01)
        self.layer2 = self.make_layer(BasicBlock1, 16, 2, stride=2)
        self.drop = nn.Dropout(0.01)
        self.layer3 = self.make_layer(BasicBlock1, 32, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class BasicBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=8):
        super(BasicBlock2, self).__init__()

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


class Net_22(nn.Module):
    def __init__(self, num_classes=10, num_groups=8):
        super(Net_22, self).__init__()

        self.in_channels = 8

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, 8)
        self.layer1 = self.make_layer(BasicBlock2, 8, 2, stride=1, num_groups=num_groups)
        self.layer2 = self.make_layer(BasicBlock2, 16, 2, stride=2, num_groups=num_groups)
        self.layer3 = self.make_layer(BasicBlock2, 32, 2, stride=2, num_groups=num_groups)
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


class BasicBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=1):
        super(BasicBlock3, self).__init__()

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


class Net_33(nn.Module):
    def __init__(self, num_classes=10, num_groups=1):
        super(Net_33, self).__init__()

        self.in_channels = 8

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, 8)
        self.layer1 = self.make_layer(BasicBlock3, 8, 2, stride=1, num_groups=num_groups)
        self.layer2 = self.make_layer(BasicBlock3, 16, 2, stride=2, num_groups=num_groups)
        self.layer3 = self.make_layer(BasicBlock3, 32, 2, stride=2, num_groups=num_groups)
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

# ---------------------------------------------------------
#           S7 Models
# ---------------------------------------------------------


class Net_1(nn.Module):
    """
     This defines the architecture or structure of the neural network.
    """

    def __init__(self):
        super(Net_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU()
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU()
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=1),
            nn.ReLU()
        )
        self.adaptive_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.relu(self.conv1x1(x))
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.adaptive_avg_pool(x)
        x = x.squeeze()
        return F.log_softmax(x, dim=1)


class Net_2(nn.Module):
    """
     This defines the architecture or structure of the neural network.
    """

    def __init__(self):
        super(Net_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU()
        )

        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU()
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=1)
        )
        self.adaptive_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)

        x = F.relu(self.conv1x1_1(x))
        x = self.conv2(x)
        # print(x.shape)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool(x)

        x = F.relu(self.conv1x1_2(x))

        x = self.conv5(x)
        x = self.conv6(x)

        x = F.relu(self.conv1x1_3(x))
        x = self.adaptive_avg_pool(x)
        x = x.squeeze()

        return F.log_softmax(x, dim=1)


class Net_4(nn.Module):
    """
     This defines the architecture or structure of the neural network.
    """

    def __init__(self):
        super(Net_4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
            # ,nn.Dropout2d(0.1)

        )
        # pool
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8)
            # nn.Dropout2d(0.1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1)
        )
        self.adaptive_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)

        x = F.relu(self.conv1x1_1(x))
        x = self.conv2(x)
        # print(x.shape)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)

        # x = self.pool(x)

        x = F.relu(self.conv1x1_2(x))

        x = self.conv5(x)
        x = self.conv6(x)

        # x = F.relu(self.conv1x1_3(x))
        x = self.adaptive_avg_pool(x)
        x = self.conv1x1_3(x)
        x = x.squeeze()

        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------
#           S7 Models
# ---------------------------------------------------------

class Net(nn.Module):
    """
     This defines the architecture or structure of the neural network.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.onecross1 = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1)

        )
        self.onecross1_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=1)
        )

        self.fc1 = nn.Sequential(
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.onecross1(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.onecross1_2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(x)
        x = x.squeeze()
        return F.log_softmax(x, dim=1)


def model_summary(model, input_size):
    """
    This function displays a summary of the model, providing information about its architecture,
    layer configuration, and the number of parameters it contains.
    :param model: model
    :param input_size: input_size for model
    :return:
    """
    summary(model, input_size=input_size)
