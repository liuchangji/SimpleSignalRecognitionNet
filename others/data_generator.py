import numpy as np
import matplotlib.pyplot as plt
from random import choice
import torch

def sin_point_generator(show=False):
    label = 0
    x = np.linspace(0, 2 * np.pi, 100)  # 8/2=4 4个完整的正弦波波形   8*16代表一共生成的点数量
    y = np.sin(x)  # 调整43这个值可以调整波峰值

    if show:
        plt.plot(x, y, 'bp--')  # 绘制成图表
        plt.show()  # 显式图像
    return y, label


def line_point_generator(show=False):
    label = 1

    x = np.linspace(0, 2 * np.pi, 100)  # 8/2=4 4个完整的正弦波波形   8*16代表一共生成的点数量
    y = x  # 调整43这个值可以调整波峰值

    if show:
        plt.plot(x, y, 'bp--')  # 绘制成图表
        plt.show()  # 显式图像
    return y, label


def curve_point_generator(show=False):
    label = 2

    x = np.linspace(0, 2 * np.pi, 100)  # 8/2=4 4个完整的正弦波波形   8*16代表一共生成的点数量
    y = np.sin(x) + 0.5 * x  # 调整43这个值可以调整波峰值

    if show:
        plt.plot(x, y, 'bp--')  # 绘制成图表
        plt.show()  # 显式图像
    return y, label


def complex_line_generator(show=False):
    label = 3
    x1 = np.linspace(0, np.pi, 50)
    y1 = np.sin(x1)
    x2 = np.linspace(np.pi, 2 * np.pi, 50)
    y2 = x2 - 3

    y = np.r_[y1, y2]
    x = np.linspace(0, np.pi, 100)

    if show:
        plt.plot(x, y, 'bp--')  # 绘制成图表
        plt.show()  # 显式图像

    return y, label


def random_choice_data():
    """
    :return: 随机选择一个数据生成器
    """
    data_generator = [complex_line_generator, curve_point_generator, line_point_generator, sin_point_generator]

    data, lable = choice(data_generator)()

    data = torch.tensor(torch.from_numpy(data), dtype=torch.float32)

    data = data.unsqueeze(0).unsqueeze(0)  # 将一个shape=[n] 升维度为[1,1,n],对应的参数为[batch,channel,number of point]

    lable = torch.tensor([lable])
    return data, lable

def save_to_txt(y,lable):

    with open("./data/"+str(lable)+".txt", "a") as f:
        for i in y:
            f.write(str(i)+"\n")


if __name__ == "__main__":

    y, lable = sin_point_generator(True)
    # save_to_txt(y, lable)
    y, lable = line_point_generator(True)
    # save_to_txt(y, lable)
    y, lable = curve_point_generator(True)
    # save_to_txt(y, lable)
    y, lable = complex_line_generator(True)
    # save_to_txt(y, lable)
