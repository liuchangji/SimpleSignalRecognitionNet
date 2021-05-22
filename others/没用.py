import torch
from torch import nn
import numpy as np
import torchvision.transforms as t
from PIL import Image
import cv2
if __name__ == "__main__":

    pass
    # loss = nn.CrossEntropyLoss()
    # # input = torch.tensor([[1.,2.,3.]])
    # # target = torch.tensor([0])
    # # output = loss(input, target)
    # #
    # # print(input)
    # #
    # # bn = nn.BatchNorm1d(1)
    # #
    # # bn(input)
    # #
    # # print(bn)
    #
    #
    # bn = nn.BatchNorm1d(33, affine=False)
    #
    # m = nn.Conv1d(16, 33, 3, stride=2)
    # input = torch.randn(20, 16, 50)
    # out= m(input)
    # out = bn(out)
    #
    # a = []
    #
    # a.append(1)
    # a.append(2)
    # print(a)
    # path = "../data/simple_signal/0.png"
    # img = cv2.imread(path)
    # print(0)
    a = np.array([[100,200,300,400,500]])
    img = Image.fromarray(a.astype('float'))
    img = img.resize((12,1),Image.NEAREST)
    img = np.array(img)
    print(img)

