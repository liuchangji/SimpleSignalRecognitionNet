# SSR-NET:SimpleSignalRecognitionNet
# 简单信号识别网络
中科院长春光学精密机械与物理研究所 图像处理技术研究部 刘长吉
## 程序结构
data_loader.py:数据集加载器

detect.py:检测器

model.py:网络模型

train.py:训练器

val.py:评估器
## 如何使用

### 0.数据集放置格式:

数据集遵循以下格式:
在你的数据文件夹内创建train与val.一个放置训练数据,另外一个放置验证数据

train文件夹内创建0,1,2,3(假设你有4个类)等等文件夹,文件夹的名字就是数据标签.
类别编号一定要从0开始,如果你有5个类,那么就应该有5个文件夹0,1,2,3,4

0,1,2,3文件夹中放置数据文件*．txt,数量不限名字随意.注意文件夹0,1,2,3是他的标签
可以参考data/simple_signal/中的放置格式


### 1.数据txt文件


txt文件名随意,放在正确的文件夹
txt每一行为一个采样时间单位
一行可以有多个数据,代表多个传感器,用空格隔开

### 2.设置config.yaml文件
在config文件夹中创建你的配置文件(参考my_net.yaml)
nc: 4  # 类别数量

input_channel: 1  输入数据的维度

data_pick_index: [0] # 选你数据集txt中哪几列数据,从0开始,如果你只有一列的数据,那就填0.所以你选取几列的数量必须等于input_channel

data_length: 100 # 输入数据长度,不等于这个的会被缩放,最近邻插值

train_dataset : "./data/simple_signal/train" 训练集路径

val_dataset : "./data/simple_signal/val" 验证集路径

weight_save_name : "my_net.pth" #权重保存名字

### 3.加载配置文件
打开config/read_config.py
修改cfgfile = "./config/my_net.yaml"为你的路径

### 4.训练
配置文件设置号后
使用train.py
设置权重保存路径:WEIGHTS_SAVE_PATH = './weights/'  
设置TRAINING_EPOCH
设置TRAINING_BATCH_SIZE
选择网络(可以在model.py中自己修改或添加):

net = SignalRecognitionNet_v3(input_channel=INPUT_CHANNEL, output_class=OUTPUT_CLASS)

OK开始训练


### 5.验证
使用val.py,和train没什么区别,设置一下权重路径就OK

### 6.检测
使用detect.py
设置file_path即可

　
