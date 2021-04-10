from pytorch.CIFAR10.model_1 import Bottleneck
import torch.nn as nn
import torch


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 256 * 3 * 32 * 32
        self.pool1 = nn.MaxPool2d(2, 2)  # 256 * 64 * 32 * 32
        self.res1 = Bottleneck(64, 64)  # 256 * 64 * 16 * 16
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # 256 * 256 * 16 * 16
        self.pool2 = nn.MaxPool2d(2, 2)  # 256 * 512 * 16 * 16
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)  # 256 * 512 * 8 * 8
        self.pool3 = nn.MaxPool2d(2, 2)  # 256 * 1024 * 8 * 8
        self.conv4 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)  # 256 * 1024 * 4 * 4
        self.pool4 = nn.MaxPool2d(2, 2)  # 256 * 1024 * 2 * 2
        self.fc3 = nn.Linear(1024 * 1 * 1, 512)  # 256 * 1024 * 1 * 1
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 10)

    def forward(self, x):

        out = self.pool1(torch.relu(self.conv1(x)))
        out = self.res1(out)
        out = self.pool2(torch.relu(self.conv2(out)))
        out = self.pool3(torch.relu(self.conv3(out)))
        out = self.pool4(torch.relu(self.conv4(out)))
        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        out = torch.relu(self.fc5(out))
        out = self.fc6(out)
        return out

