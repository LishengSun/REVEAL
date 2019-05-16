import torch
import torch.nn as nn
import torch.nn.functional as F


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEGMENT_LENGTH=30

class navigation_model(nn.Module):
    def __init__(self):
        super(navigation_model, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(64 * SEGMENT_LENGTH, SEGMENT_LENGTH)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear1(x)
        # print(x.shape)
        x = x.view(x.size(0), 1, -1)
        # print(x.shape)
        return x


class big_navigation_model(nn.Module):
    def __init__(self):
        super(big_navigation_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2, 3), padding=(0, 1))
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2, return_indices=True)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.conv9 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.conv8 = nn.Conv1d(16, 1, kernel_size=3, padding=1)
        # self.linear1 =
        # self.linear1 = nn.Linear(64*SEGMENT_LENGTH, SEGMENT_LENGTH)

    def forward(self, x):
        size0 = x.size()
        # print(x.shape)
        x = x.view(x.size(0), 1, 2, -1)
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), x.size(1), -1)
        x = F.relu(self.conv2(x))
        x, ind1 = self.pool(x)
        size1 = x.size()
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x, ind2 = self.pool(x)
        size2 = x.size()
        # print(x.shape)
        x = F.relu(self.conv4(x))
        # print(x.shape)
        x, ind3 = self.pool(x)
        # print(x.shape)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv9(x))
        # print(x.shape)
        x = self.unpool(x, ind3)
        # print(x.shape)
        x = F.relu(self.conv6(x))
        # print(x.shape)
        x = self.unpool(x, ind2)
        # print(x.shape)
        x = F.relu(self.conv7(x))
        # print(x.shape)
        x = self.unpool(x, ind1)
        # print(x.shape)
        x = self.conv8(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        # x = self.linear1(x)
        # print(x.shape)
        # x = x.view(x.size(0), 1, -1)
        return x


class assembled_model(nn.Module):
    def __init__(self, oracle=True):
        super(assembled_model, self).__init__()

    def forward(self, s):
        # print(s.shape)
        x = navigation_model(s)
        # print(x.shape)
        return x
