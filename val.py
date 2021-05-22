import torch
from model import SignalRecognitionNet_v3
from others.data_generator import random_choice_data

import time
from data_loader import Signaldata
from torch.utils.data import DataLoader
from config.read_config import parse_cfg

cfg = parse_cfg()
# 读取训练参数
TRAINING_DATA_PATH = cfg["train_dataset"]
VAL_DATA_PATH = cfg["val_dataset"]
DATA_LENGTH = cfg["data_length"]
INPUT_CHANNEL = cfg["input_channel"]
OUTPUT_CLASS = cfg["nc"]
PICK_INDEX = cfg["data_pick_index"]
WEIGHT_SAVE_NAME = cfg["weight_save_name"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = SignalRecognitionNet_v3(input_channel=2, output_class=3).to(device)  # 初始化网络
PATH = './weights/my_net-best.pth'  # 权重保存路径
net.load_state_dict(torch.load(PATH))  # 加载权重
VAL_DATA = Signaldata(VAL_DATA_PATH, pick_index=PICK_INDEX, data_length=DATA_LENGTH, val=True)


conf_thresh = 0.5
point = 0  # 计分器
net.eval()
with torch.no_grad():
    for epoch in range(1):  #
        test_dataloder = DataLoader(VAL_DATA, batch_size=1,
                                    num_workers=0, drop_last=True, shuffle=True)

        for step, data in enumerate(test_dataloder):
            input_data = data[0].type(torch.FloatTensor).to(device)
            lable = data[1].to(device)

            start_time = time.time()
            outputs = net(input_data.to(device))
            end_time = time.time()
            spend_time = end_time - start_time
            FPS = 1 / spend_time

            outputs = outputs.cpu().detach().numpy()[0, :]
            outputs = outputs.tolist()

            if max(outputs) < conf_thresh:
                continue

            target_class = outputs.index(max(outputs))  # 返回最大值的索引

            lable = lable.cpu().numpy().tolist()[0]

            if target_class == lable:
                point = point + 1

print('准确率', round(point / VAL_DATA.__len__(), 4) * 100, "%")

