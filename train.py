import time
import torch
import torch.nn as nn
from model import SignalRecognitionNet_v1, SignalRecognitionNet_v3, SignalRecognitionNet_v4
from data_loader import Signaldata
from torch.utils.data import DataLoader
from config.read_config import parse_cfg
import os

cfg = parse_cfg()
# 读取训练参数
TRAINING_DATA_PATH = cfg["train_dataset"]
VAL_DATA_PATH = cfg["val_dataset"]
DATA_LENGTH = cfg["data_length"]
INPUT_CHANNEL = cfg["input_channel"]
OUTPUT_CLASS = cfg["nc"]
PICK_INDEX = cfg["data_pick_index"]
WEIGHT_SAVE_NAME = cfg["weight_save_name"]  # 权重保存名字

# set your save path
WEIGHTS_SAVE_PATH = './weights/'  # 权重保存路径
WEIGHT_SAVE_ID = WEIGHT_SAVE_NAME.split(".")[0]
BEST_WEIGHTS_SAVE_PATH = os.path.join(WEIGHTS_SAVE_PATH, WEIGHT_SAVE_ID + "-best.pth")  # 最佳权重保存路径
WEIGHTS_SAVE_PATH = os.path.join(WEIGHTS_SAVE_PATH, WEIGHT_SAVE_NAME)

# read data
TRAIN_DATA = Signaldata(TRAINING_DATA_PATH, pick_index=PICK_INDEX, data_length=DATA_LENGTH)  # 加载数据集
VAL_DATA = Signaldata(VAL_DATA_PATH, pick_index=PICK_INDEX, data_length=DATA_LENGTH, val=True)
TRAINING_EPOCH = 400  # 训练epoch
TRAINING_BATCH_SIZE = 10  #

if TRAINING_BATCH_SIZE > TRAIN_DATA.__len__():
    print("BATCH_SIZE超过训练集数量,修改为TRAINING_BATCH_SIZE = ", TRAIN_DATA.__len__())
    TRAINING_BATCH_SIZE = TRAIN_DATA.__len__()

# 开始
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# choose your net
net = SignalRecognitionNet_v3(input_channel=INPUT_CHANNEL, output_class=OUTPUT_CLASS).to(device)

compute_loss = nn.CrossEntropyLoss()  # 使用交叉熵函数

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
conf_thresh_for_val = 0.5  # 用于验证的置信度,与训练无关

precision = 0

for epoch in range(TRAINING_EPOCH):
    net.train()
    running_loss = 0.0

    # get the inputs

    optimizer.zero_grad()  # zero the parameter gradients

    train_dataloder = DataLoader(TRAIN_DATA, batch_size=TRAINING_BATCH_SIZE,
                                 num_workers=0, drop_last=True, shuffle=True)

    for step, data in enumerate(train_dataloder):
        input_data = data[0].type(torch.FloatTensor).to(device)
        lable = data[1].to(device)
        file_name_for_check = data[2]
        outputs = net(input_data)

        loss = compute_loss(outputs, lable)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:  # 每10个epoch评估一次
        outputs = outputs.cpu().tolist()[0]
        outputs = [round(i, 3) for i in outputs]
        print(" ")
        print('current_epoch={} output={} loss={}'.format(epoch, outputs, round(loss.cpu().tolist(), 5)))
        # 训练中评估
        print('评估')
        point = 0
        with torch.no_grad():
            net.eval()
            test_dataloder = DataLoader(VAL_DATA, batch_size=1,
                                        num_workers=0, drop_last=True, shuffle=True)

            for step, data in enumerate(test_dataloder):
                input_data = data[0].type(torch.FloatTensor).to(device)
                lable = data[1].to(device)
                file_name_for_check = data[2]
                start_time = time.time()
                outputs = net(input_data.to(device))
                end_time = time.time()
                spend_time = end_time - start_time
                FPS = 1 / spend_time

                outputs = outputs.cpu().detach().numpy()[0, :]
                outputs = outputs.tolist()

                if max(outputs) < conf_thresh_for_val:
                    continue

                target_class = outputs.index(max(outputs))  # 返回最大值的索引

                lable = lable.cpu().numpy().tolist()[0]

                if target_class == lable:
                    point = point + 1
            p = round(point / VAL_DATA.__len__(), 4) * 100
            print('准确率', p, "%")
            if p > precision:
                torch.save(net.state_dict(), BEST_WEIGHTS_SAVE_PATH)  # SAVE WEIGHT
                precision = p

print('结束训练\n')

torch.save(net.state_dict(), WEIGHTS_SAVE_PATH)  # SAVE WEIGHT

print('最终评估')
point = 0  # 计分器
with torch.no_grad():
    net.eval()

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

        if max(outputs) < conf_thresh_for_val:
            continue

        target_class = outputs.index(max(outputs))  # 返回最大值的索引

        lable = lable.cpu().numpy().tolist()[0]

        if target_class == lable:
            point = point + 1

print('准确率', round(point / VAL_DATA.__len__(), 4) * 100, "%")

if __name__ == "__main__":
    pass
    # x_target = torch.tensor([1,0,0]).unsqueeze(0)
    #
    # y_target = torch.tensor([1])
    #
    # a=nn.CrossEntropyLoss()
    #
    # print(a(x_target,y_target))
