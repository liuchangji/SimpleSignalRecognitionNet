# 数据集构建 https://zhuanlan.zhihu.com/p/105507334
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image

class Signaldata(Dataset):  # 需要继承data.Dataset
    def __init__(self, path, pick_index,data_length,val=False):
        """
        :param path:数据集路径
        :param data_length:数据长度resize,不是 data_length的会缩放成 data_length
        :param pick_index:选取某些维度索引,用来选取你需要使用第几维度的数据 for example (0,1,2,3,4)
        """
        self.pick_index = pick_index
        self.path = path
        self.data_length = data_length
        self.lable = os.listdir(path)  # 获得文件夹名字
        self.lable_and_data_file_name = []  # 创建一个保存标签与数据文件名的list
        for i in self.lable:
            path_ = os.path.join(path, i)  # 打开每个标签文件夹
            file_name_list = os.listdir(path_)  # 获得标签文件夹内所有数据文件名
            for j in file_name_list:
                self.lable_and_data_file_name.append((i, j))  # 把标签和对应的文件名的存在list中

        self.file_number = len(self.lable_and_data_file_name)  # 返回数据集数量
        if val==False:
            print("训练数据集数量:", self.file_number)
        else:
            print("验证数据集数量:", self.file_number)


    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        file_name = self.lable_and_data_file_name[index][1]
        its_lable = self.lable_and_data_file_name[index][0]
        file_path = os.path.join(self.path, its_lable, file_name)

        with open(file_path, "r") as f:
            original_data = f.read().splitlines()  # 打开数据文件
        data_after_filter = []
        for index, item in enumerate(original_data):
            content = item.split(" ")
            content_single_line = []
            for pick in self.pick_index:
                content_single_line.append(float(content[pick]))
            data_after_filter.append(content_single_line)
            # data_after_filter.append([float(content[0])])

        data = np.array(data_after_filter).T

        data_ = np.zeros((len(self.pick_index),self.data_length))

        # 对数据进行缩放,找了很多,只有PIL库能缩放一维数据
        for i in range(len(self.pick_index)):

            temp = data[i,:]
            temp = temp[None,:]
            temp = Image.fromarray(temp.astype('float'))
            temp = temp.resize((self.data_length,1), Image.NEAREST)
            temp = np.array(temp)

            data_[i,:] = temp[0]



        # data = torch.Tensor(data)
        # data = data.resize(len(self.pick_index),self.data_length)
        return data_, int(its_lable), file_name

    def __len__(self):
        #:return: 返回数据集长度

        return self.file_number


if __name__ == "__main__":
    path = "./data/multi-angle/train"

    dataset = Signaldata(path, pick_index=[1, 2],data_length=800)  # 加载数据集


    train_dataloder = DataLoader(dataset, batch_size=2,
                                 num_workers=0, drop_last=True, shuffle=True)
    for index, data in enumerate(train_dataloder):
        print(data)

    # with open(path, "r") as f:
    #     original_data = f.read().splitlines()
    # data_after_filter = []
    # for index, item in enumerate(original_data):
    #     content = item.split(" ")
    #     data_after_filter.append([float(content[1]), float(content[2])])
    #
    # data = np.array(data_after_filter).T
    #
    # print(data)

    # train_dataset = Signaldata(path)
    # train_dataloder = DataLoader(train_dataset, batch_size=2,
    #                              num_workers=0, drop_last=True, shuffle=True)
    #
    #
    #
    #
    #
    #
    # for step, data in enumerate(train_dataloder):
    #     print(data[0])
