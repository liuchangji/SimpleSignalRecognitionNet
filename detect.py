import torch
from model import SignalRecognitionNet_v3
import time
import os
import numpy as np
from config.read_config import parse_cfg
from PIL import Image
cfg = parse_cfg()
DATA_LENGTH = cfg["data_length"]
INPUT_CHANNEL = cfg["input_channel"]
OUTPUT_CLASS = cfg["nc"]
PICK_INDEX = cfg["data_pick_index"]  # 选取数据的第几列作为输入,如果你的数据是五个维度,全部需要的话就是[0,1,2,3,4]

RESIZE = True  # 是否根据你的 DATA_LENGTH resize你的输入数据

WEIGHT_SAVE_NAME = cfg["weight_save_name"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = SignalRecognitionNet_v3(input_channel=INPUT_CHANNEL, output_class=OUTPUT_CLASS).to(device)  # 初始化网络

PATH = './weights/SignalRecognitionNet-v3-hardsignal.pth'  # 读取权重保存路径

net.load_state_dict(torch.load(PATH))  # 加载权重

file_path = "./data/hard_signal/val/0/2KM_1.37_missile1_save040.txt"  # 数据路径

with open(file_path, "r") as f:
    original_data = f.read().splitlines()  # 打开数据文件

data_after_filter = []
for index, item in enumerate(original_data):
    content = item.split(" ")
    content_single_line = []
    for pick in PICK_INDEX:
        content_single_line.append(float(content[pick]))
    data_after_filter.append(content_single_line)
    # data_after_filter.append([float(content[0])])

data = np.array(data_after_filter).T

if RESIZE:
    data_ = np.zeros((len(PICK_INDEX), DATA_LENGTH))

    # 对数据进行缩放,找了很多,只有PIL库能缩放一维数据
    for i in range(len(PICK_INDEX)):
        temp = data[i, :]
        temp = temp[None, :]
        temp = Image.fromarray(temp.astype('float'))
        temp = temp.resize((DATA_LENGTH, 1), Image.NEAREST)
        temp = np.array(temp)
        data_[i, :] = temp[0]
    data = data_

data = torch.tensor((data))

data = data.unsqueeze(0).type(torch.FloatTensor)  #

with torch.no_grad():
    input_data = data
    net.eval()
    outputs = net(input_data.to(device))
    outputs = outputs.cpu().detach().numpy()[0, :]
    outputs = outputs.tolist()
    target_class = outputs.index(max(outputs))  # 返回最大值的索引
    print("信号的类别为:{}  置信度为:{}".format(target_class, round(max(outputs), 3)))
