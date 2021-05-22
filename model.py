import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalRecognitionNet_v1(nn.Module):
    def __init__(self, input_channel, output_class):
        super(SignalRecognitionNet_v1, self).__init__()


        self.conv1 = nn.Conv1d(input_channel, 4, 3, 1, 1)
        self.conv2 = nn.Conv1d(4, 4, 3, stride=1, padding=1)  # P0
        self.conv3 = nn.Conv1d(4, 8, 3, 2, 1)  # P1 /2
        self.conv4 = nn.Conv1d(8, 16, 3, 2, 1)  # P2 /4
        self.conv5 = nn.Conv1d(16, 32, 3, 2, 1)  # P3 /8
        self.conv6 = nn.Conv1d(32, 64, 3, 2, 1)  # P4 /16
        self.conv7 = nn.Conv1d(64, 16, 3, 2, 1)  # P5 /32

        self.roi_pooling = nn.AdaptiveMaxPool1d(16)
        self.liner1 = nn.Linear(16 * 16, output_class)

    def activation_function(self, x):
        return torch.tanh(x)

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        o = self.conv1(x)

        o = self.activation_function(o)
        o = self.conv2(o)
        o = self.activation_function(o)
        o = self.conv3(o)
        o = self.activation_function(o)
        o = self.conv4(o)
        o = self.activation_function(o)
        o = self.conv5(o)
        o = self.activation_function(o)
        o = self.conv6(o)
        o = self.activation_function(o)
        o = self.conv7(o)
        o = self.activation_function(o)
        o = self.roi_pooling(o)
        o = self.activation_function(o)
        o = o.view(-1, 16 * 16)  # 展开为一维向量连接全连接层
        o = self.liner1(o)
        o = F.softmax(o)
        return o


class SignalRecognitionNet_v2(nn.Module):
    def __init__(self, input_channel, output_class):
        super(SignalRecognitionNet_v2, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 4, 3, 1, 1)
        self.conv2 = nn.Conv1d(4, 4, 3, stride=1, padding=1)  # P0
        self.conv3 = nn.Conv1d(4, 8, 3, 2, 1)  # P1 /2
        self.conv4 = nn.Conv1d(8, 16, 3, 2, 1)  # P2 /4
        self.conv5 = nn.Conv1d(16, 32, 3, 2, 1)  # P3 /8
        self.conv6 = nn.Conv1d(32, 64, 3, 2, 1)  # P4 /16
        self.conv7 = nn.Conv1d(64, 16, 3, 2, 1)  # P5 /32
        self.roi_pooling = nn.AdaptiveMaxPool1d(4)
        self.liner1 = nn.Linear(4 * 16, output_class)

    def activation_function(self, x):
        return torch.relu(x)

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        o = self.conv1(x)
        o = self.activation_function(o)
        o = self.conv2(o)
        o = self.activation_function(o)
        o = self.conv3(o)
        o = self.activation_function(o)
        o = self.conv4(o)
        o = self.activation_function(o)
        o = self.conv5(o)
        o = self.activation_function(o)
        o = self.conv6(o)
        o = self.activation_function(o)
        o = self.conv7(o)
        o = self.activation_function(o)
        o = self.roi_pooling(o)
        o = self.activation_function(o)
        o = o.view(1, -1)  # 展开为一维向量连接全连接层
        o = self.liner1(o)
        o = F.softmax(o)

        return o

class SignalRecognitionNet_v3(nn.Module):
    def __init__(self, input_channel, output_class):
        """
        The best
        :param input_channel:
        :param output_class:
        """
        super(SignalRecognitionNet_v3, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 4, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(4, 4, 3, stride=1, padding=1)  # P0
        self.bn2 = nn.BatchNorm1d(4)
        self.conv3 = nn.Conv1d(4, 8, 3, 2, 1)  # P1 /2
        self.bn3 = nn.BatchNorm1d(8)
        self.conv4 = nn.Conv1d(8, 16, 3, 2, 1)  # P2 /4
        self.bn4 = nn.BatchNorm1d(16)
        self.conv5 = nn.Conv1d(16, 32, 3, 2, 1)  # P3 /8
        self.bn5 = nn.BatchNorm1d(32)
        self.conv6 = nn.Conv1d(32, 64, 3, 2, 1)  # P4 /16
        self.bn6 = nn.BatchNorm1d(64)
        self.conv7 = nn.Conv1d(64, 16, 3, 2, 1)  # P5 /32
        self.bn7 = nn.BatchNorm1d(16)
        self.roi_pooling = nn.AdaptiveAvgPool1d(16)

        # self.roi_pooling = nn.AdaptiveMaxPool1d(16)
        self.liner1 = nn.Linear(16 * 16, output_class)

    def activation_function(self, x):
        return torch.relu(x)

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.activation_function(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.activation_function(o)
        o = self.conv3(o)
        o = self.bn3(o)
        o = self.activation_function(o)
        o = self.conv4(o)
        o = self.bn4(o)
        o = self.activation_function(o)
        o = self.conv5(o)
        o = self.bn5(o)
        o = self.activation_function(o)
        o = self.conv6(o)
        o = self.bn6(o)
        o = self.activation_function(o)
        o = self.conv7(o)
        o = self.bn7(o)
        o = self.activation_function(o)

        o = self.roi_pooling(o)
        o = self.activation_function(o)
        o = o.view(-1, 16 * 16)  # 展开为一维向量连接全连接层
        o = self.liner1(o)
        o = F.softmax(o,1)
        return o

class SignalRecognitionNet_v4(nn.Module):
    """
    HighResolution
    """
    def __init__(self, input_channel, output_class):

        super(SignalRecognitionNet_v4, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 4, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(4, 4, 3, stride=1, padding=1)  # P0
        self.bn2 = nn.BatchNorm1d(4)
        self.conv3 = nn.Conv1d(4, 8, 3, 2, 1)  # P1 /2
        self.bn3 = nn.BatchNorm1d(8)
        self.conv4 = nn.Conv1d(8, 16, 3, 1, 1)  # P2 /4
        self.bn4 = nn.BatchNorm1d(16)
        self.conv5 = nn.Conv1d(16, 32, 3, 2, 1)  # P3 /8
        self.bn5 = nn.BatchNorm1d(32)
        self.conv6 = nn.Conv1d(32, 64, 3, 1, 1)  # P4 /16
        self.bn6 = nn.BatchNorm1d(64)
        self.conv7 = nn.Conv1d(64, 32, 3, 2, 1)  # P5 /32
        self.bn7 = nn.BatchNorm1d(32)

        self.roi_pooling = nn.AdaptiveMaxPool1d(32)
        self.liner1 = nn.Linear(32 * 32, 32*16)
        self.liner2 = nn.Linear(32 * 16, 16 * 16)
        self.liner3 = nn.Linear(16 * 16, 16 * 8)
        self.liner4 = nn.Linear(128,output_class)

    def activation_function(self, x):
        return torch.nn.functional.tanh(x)

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.activation_function(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.activation_function(o)
        o = self.conv3(o)
        o = self.bn3(o)
        o = self.activation_function(o)
        o = self.conv4(o)
        o = self.bn4(o)
        o = self.activation_function(o)
        o = self.conv5(o)
        o = self.bn5(o)
        o = self.activation_function(o)
        o = self.conv6(o)
        o = self.bn6(o)
        o = self.activation_function(o)
        o = self.conv7(o)
        o = self.bn7(o)
        o = self.activation_function(o)

        o = self.roi_pooling(o)
        o = self.activation_function(o)
        o = o.view(-1, 32 * 32)  # 展开为一维向量连接全连接层
        o = self.liner1(o)
        o = self.liner2(o)
        o = self.liner3(o)
        o = self.liner4(o)
        o = F.softmax(o)
        return o
if __name__ == "__main__":

    a = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 使用虚拟数据进行正向传播测试
    input = torch.randn(1, 1, 8)
    net = SignalRecognitionNet_v1(1, 5)
    output = net(input)

