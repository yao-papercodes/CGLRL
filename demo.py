# import os
# import sys
# import time
# import tqdm
# import shutil
# import argparse
# import subprocess
# import numpy as np
# import pandas as pd

# import sklearn
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE

# from PIL import Image, ImageDraw

# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import torch.utils.data as data
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms

# from Utils import *
# from Dataset import MyDataset
# from collections import namedtuple
# from VGG import VGG, VGG_onlyGlobal
# from ResNet import IR, IR_onlyGlobal
# from MobileNet import MobileNetV2, MobileNetV2_onlyGlobal
# from AdversarialNetwork import RandomLayer, AdversarialNetwork, calc_coeff

# import time
# import datetime

# ary=[1,2,3,4,5]
# print(ary[3:4])

# acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]
# print(os.getcwd())
# pre_add='../Dataset'
# labels=pd.read_csv(os.path.join(pre_add,'RAF/RAF/basic/EmoLabel/list_patition_label.txt'))
# # labels=pd.read_csv('/home/scau/桌面/XieYuhao/GCN/CD-FER-Benchmark-master/Dataset/RAF/RAF/basic/EmoLabel/list_patition_label.txt',header=None,delim_whitespace=True)
# labels=np.array(labels)
# print(labels[3,0])


# student=namedtuple('student','id age name')
# s=student(id='123',age='12',name='xieyuhao')
# print(s.id)

# a=[123,213,13]
# b=a+[12]
# print(b)

# for i in range(2):
#     print(i)
# a=[7 for i in range(7)]
# print(a)

# tensor=torch.zeros((64,512,7,7))
# conv=nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
# adj=nn.AdaptiveAvgPool2d((1,1))
# tensor=adj(tensor)
# print(tensor.shape)

# m1=torch.ones((12,12))
# m2=torch.randn((12,12))
# m1.data.copy_(m1*m2)
# print(m1)
# m3=m1*m2
# print(m3.shape)
# print(m3)

# batch1=torch.randn(10,2,6)
# batch2=torch.randn(10,6,2)
# print((torch.bmm(batch1,batch2)).shape)
# import numpy as np
# # t=torch.randn(64,384)
# # t2=torch.randn(64,384)
# # ary=[t,t2]
# # print(np.vstack(ary).shape)
# a=torch.randn(1,7)
# b=torch.randn(64,7)
# ary=[a,b]
# print(np.concatenate(ary).shape)


# coding:utf-8
# import matplotlib.pyplot as plt
# labels='apple','banana','orange','pear'
# sizes=20,10,30,40
# colors='yellowgreen','gold','lightskyblue','lightcoral'
# #banana和其它有间隙
# explode=0,0.1,0,0
# plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%.1f%%',shadow=True,startangle=50)
# #轴对称，圆
# plt.axis('equal')
# plt.show()


# S_X1=np.random.random((200,200))#原始特征
#
# class_num=10
# #####生成标签
# y_s=np.zeros((20*class_num)) # shape是1*200
# for i in range(20 * class_num):
#     y_s[i] = i // 20 # 整除
# ###################
#
# maker=['o','v','^','s','p','*','<','>','D','d','h','H']#设置散点形状
# colors = ['black','tomato','yellow','cyan','blue', 'lime', 'r', 'violet','m','peru','olivedrab','hotpink']#设置散点颜色
#
# Label_Com = ['S-1', 'T-1', 'S-2', 'T-2', 'S-3',
#              'T-3', 'S-4', 'T-4','S-5','T-5', 'S-6', 'T-6', 'S-7','T-7','S-8', 'T-8','S-9','T-9',
#              'S-10','T-10','S-11', 'T-11', 'S-12','T-12'] ##图例名称
#
# ### 设置字体格式
# font1 = {'family' : 'Times New Roman',
#
#          'weight' : 'bold',
# 'size'   : 32,
# }
#
#
# def visual(X):
#     tsne = TSNE(n_components=2,init='pca', random_state=501) # random_state是第501号随机种子
#     X_tsne = tsne.fit_transform(X)
#
#     print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
#
#     #'''嵌入空间可视化'''
#     x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#     '''
#     x_min的作用:
#     对比X_tsne所有行中的第0列，第1列的中每一列的数据，取一个最小值
#
#     '''
#     X_norm = (X_tsne - x_min) / (x_max - x_min) # x_min会自动进行广播拓展
#
#     return  X_norm
#
#
# def plot_with_labels(S_lowDWeights,Trure_labels,name):
#     plt.cla()#清除当前图形中的当前活动轴,所以可以重复利用
#
#     # 降到二维了，分别给x和y
#     True_labels=Trure_labels.reshape((-1,1)) # S_lowDWeight的shape是200*2，True_labels的shape是200*1
#
#     S_data=np.hstack((S_lowDWeights,True_labels))
#     S_data=pd.DataFrame({'x':S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})
#
#
#     for index in range(class_num):
#         # 这两步是用来获取index对应的所有点的x，y坐标
#         X= S_data.loc[S_data['label'] == index]['x']
#         Y=S_data.loc[S_data['label'] == index]['y']
#         # 画图
#         plt.scatter(X,Y,cmap='brg', s=100, marker=maker[index], c='', edgecolors=colors[index],alpha=0.65)
#
#     plt.xticks([])  # 去掉横坐标值
#     plt.yticks([])  # 去掉纵坐标值
#     #
#     plt.title(name,fontsize=32,fontweight='normal',pad=20)
#
#
#
#
# fig = plt.figure(figsize=(10,10))
# ax1 = fig.add_subplot(111) # 表示将这个画布按照1行1列进行划分，然后这张图放在第一个位置上
# plot_with_labels(visual(S_X1),y_s,'(a)')
#
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.8, wspace=0.1, hspace=0.15) # 调整子布局
# plt.legend(scatterpoints=1,labels = Label_Com, loc='best',labelspacing=0.4,columnspacing=0.4,markerscale=2,bbox_to_anchor=(0.9, 0),ncol=12,prop=font1,handletextpad=0.1)
#
# plt.savefig('./'+'TSNE2.png', format='png',dpi=300, bbox_inches='tight')
# plt.show() # 不用写成plt.show(fig),这样会报错

# tensor = torch.ones(2, 2, 2)
# tensor[0, 1, 1] = 10
# print(tensor)
# adt = nn.AdaptiveAvgPool2d((1, 1))
# tensor_ = adt(tensor)
# print(tensor_)


# x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(x.narrow(0, 0, 2))

# t = time.time()
# print(int(t))
# path = '.'
# subpath = 'data'
# print(os.path.join(path, subpath))

# a = [1, 2, 3]
# a += [4, 5]
# print(a)

import torch.nn as nn
# a = nn.ModuleList([nn.Linear(7, 64) for i in range(5)])
# a += nn.ModuleList([nn.Linear(320, 7), nn.Linear(384, 7)])
# print(a)

# [print("hello world") for i in range(5)]
# a = []
# a += [1]
# a += [2]
# print(a)

# a = 1
# [a += 2 for i in range(3)]
# print(a)

# arr = [[i for i in range(7)] for i in range(7)]
# print(arr)
# for i in range(5):
#     print(i)

# def change(arr):
#     for i in range(10):
#         arr[i] = i

# arr = [[0 for i in range(10)] for i in range(10)]
# change(arr[2])
# print(arr)

# a = [i * 12 for i in range(10)]
# string = '''
# {arr[0]}, {arr[1]}, {arr[4]}
# '''.format(arr=a)
# print(string)
# import random
# a = [random.uniform(0, 1) for i in range(7)]
# classifier_name = {0:'global', 1:'left_eye', 2:'right_eye', 3:'nose', 4:'left_mouth', 5:'right_mouth', 6:'global_local'}
# print(a)
# print(classifier_name[a.index(max(a))])

# num = 1.242412541
# print(f"number is {num:.4f}")
import time
import torch
import torch.nn as nn
import torch.optim as optim

# feature = torch.ones(32, 384)
# features = [feature.narrow(1, i*64, 64) for i in range(6)]
# fc = nn.Linear(64, 7)
# pred = fc(features[0])
# print(pred.shape)

# a = 100
# string = f"num = {a}"
# print(string)
# arr = [2, 4, 5]
# print(arr/3)

# string = 'asnda\n'\
# 'dasdassadadasdasd\n'
# print(string)

# from tqdm import tqdm
# bar = tqdm(range(100))
# for step, i in enumerate(bar):
#     time.sleep(0.25)
#     bar.desc = f'epoch {i}'

# def xie():
#     return [{"params":12}]

# arr = []
# for i in range(3):
#     arr += xie()
# print(arr)


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(64, 7)
#         self.fc2 = nn.Linear(7, 2)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x
    
#     def get_parameters(self):
#         return [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]


# model1 = Model()
# model2 = Model()

# param_groups = []
# param_groups += model1.get_parameters()
# param_groups += model2.get_parameters()
# optimizer = optim.SGD(param_groups, lr=0.1, weight_decay=1e-4, momentum=0.9)

# # print(f"训练前的参数")
# # for k, v in model2.named_parameters():
# #     # print(str(k) + '\n' + str(v) + '\n')
# #     print(str(k))

# # x = torch.randn(32, 64)
# # z = model1(x)
# # z_sum = z.sum()
# # optimizer.zero_grad()
# # z_sum.backward()
# # optimizer.step()

# # print(f"训练后的参数")
# # for k, v in model2.named_parameters():
# #     # print(str(k) + '\n' + str(v) + '\n')
# #     print(str(k))

# for model in param_groups:
#     print(model)


# a = [0, 1, 1, 1, 1, 1, 2]
# for step, i in enumerate(a):
#     print(step)
#     # print(i)
# import numpy as np
# pred = torch.randn(64, 7)
# pred = np.argmax(pred, axis=1)
# print(pred)
# import numpy as np
# a = np.array([1, 2, -2, 3, 2, 1])
# b = np.array([-1, 2, 2, 3, 2, 1])
# c = np.array([-1, 3, 4, 2, 2, 9])

# for i in range(6, 6):
#     print(i)

# a = [2, 222, 444, 123]
# print(a[1:2])

# import torch
# from torch.autograd  import  Function
 
# x = torch.tensor([1.,2.,3.],requires_grad=True)
# y = torch.tensor([4.,5.,6.],requires_grad=True)
 
# z = torch.pow(x,2) + torch.pow(y,2)
# f = z + x + y
# s =6* f.sum()
 
# print(s)
# s.backward()
# print(x)
# print(x.grad)


# import torch
# from torch.autograd  import  Function
 
# x = torch.tensor([1.,2.,3.],requires_grad=True)
# y = torch.tensor([4.,5.,6.],requires_grad=True)
 
# z = torch.pow(x,2) + torch.pow(y,2)
# f = z + x + y
 
# class GRL(Function):
#     def forward(self,input):
#         return input
#     def backward(self,grad_output):
#         grad_input = grad_output.neg()
#         return grad_input
 
 
# Grl = GRL()
 
# s =6* f.sum()
# s = Grl(s)
 
# print(s)
# s.backward()
# print(x)
# print(x.grad)
# import numpy as np
# entropy = 0
# for i in range(7):
#     entropy += -(1/7) * np.log2(1/7)
# print(entropy)

# import torch.nn as nn 
def Entropy(input_):
    return torch.sum(-input_ * torch.log(input_ + 1e-5), dim=0)

# # tensor = torch.full((32, 7), 1/7)
# # print(Entropy(tensor))

# tensor = torch.tensor([[10, 0, 0], [4, 5, 9]], dtype=torch.float)
# tensor = nn.Softmax(dim=1)(tensor)
# print(tensor)
# print(Entropy(tensor))

# tensor = torch.tensor(100.)
# print(torch.log(tensor))

# a = torch.ones(32, 7)
# b = torch.ones(32, 7)
# c = torch.ones(32, 7)
# d = 100
# if a[0][0] == b[0][0] == c[1][1]\
#     and d == 100 :
#     print("yep")

# tensor = torch.tensor([1/7 for i in range(7)])
# print(Entropy(tensor))

# import numpy as np
# print(torch.log(torch.tensor(7.)))
# cnt = 0
# arr = np.arange(1.9459/10, 1.9459+1.9459/10, 1.9459/10)
# print(arr)

# d = {'s':1}
# for i in range(1, 10):
#     d.update({'s'+str(i): i})
# print(d)

# tensor = torch.tensor([[2, 4, 6], [1, 3, 5], [7, 8, 9], [10, 11, 21]])
# indexs = [1, 3]
# print(tensor[indexs])

import torch

# tensor1 = torch.ones(1, 1, 16)
# tensor2 = torch.zeros(1, 1, 16)
# tensor3 = torch.cat((tensor1, tensor2), dim=1)
# print(tensor3.shape)
# tensor4 = tensor3.view(1, 2, -1)
# print(tensor4)

# index = torch.tensor([])
# index.append(torch.tensor(2))
# index.append(torch.tensor(3))
# index.append(torch.tensor(1))
# print(index)
# import numpy as np
# # value = np.array([[2, 3, 4, 5], [2, 32, 2, 0], [-12, -32, 23, 99]])
# # index = np.array([0, 2])
# # print(value[index])

# base_action = [ 2.         -1.65909091  2.         -2.          2.          0.22669992
#   2.         -2.         -1.62084658  2.         -0.40106834 -2.
#  -0.19870362 -1.6288685  -2.          0.02443762 -2.         -1.7053572
#   0.83902649 -1.48000453  2.         -1.44656077 -2.          2.
#  -1.47518171 -2.          0.44459582 -0.51717101  2.          2.        ]

# print(int(base_action))


# import matplotlib.pyplot as plt
# from numpy import*
# from math import*

# plt.figure(figsize=(6,6))
# wh = hh = 6/2
# t = arange(0,4*pi,0.01)
# plt.plot([0,0],[-3,3],'-r')
# plt.plot([-3,3],[0,0],'-r')
# x = (wh/2*((cos(5/2*t)**3)+sin(t))*cos(t))
# y = (hh/2*(cos(5/2*t)**3+sin(t))*sin(t))
# plt.plot(x, y, '-b')
# plt.show()
# from tqdm import tqdm
# bar = iter(range(10))
# print(bar.next())
import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 3, 4, 4, 1])
print(a == b)
print(np.sum(a == b))