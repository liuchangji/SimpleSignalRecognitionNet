import numpy as np
import os

"""
读取单个txt数据
"""

def read_one_txt(path):

    with open(path, "r") as f:
        data = f.read().splitlines()
    data = np.array([float(x) for x in data])
    return data

if __name__ == "__main__":
   path = "../data/train/1.txt"
   a=read_one_txt(path)
   print(a)