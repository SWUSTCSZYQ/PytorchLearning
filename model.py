import torch
import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 输入图片为 1 * 1 * 32 * 32
        self.conv1 = nn.Conv2d(1, 6, 3)  # 6 * 30 * 30 padding默认为0
        self.pool1 = nn.MaxPool2d(2, 2)  # 6 * 15 * 15
        self.conv2 = nn.Conv2d(6, 16, 3)  # 16 * 13 * 13
        self.pool2 = nn.MaxPool2d(2, 2)  # 16 * 6 * 6
        self.fc3 = nn.Linear(16 * 6 * 6, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):

        x = self.pool1(torch.relu((self.conv1(x))))
        x = self.pool2(torch.relu((self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x



