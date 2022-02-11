import os
import sys
import time
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

import sklearn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from PIL import Image, ImageDraw

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Dataset import MyDataset
from VGG import VGG, VGG_onlyGlobal
from ResNet import IR, IR_onlyGlobal
from MobileNet import MobileNetV2, MobileNetV2_onlyGlobal
from AdversarialNetwork import RandomLayer, AdversarialNetwork, calc_coeff
class AverageMeter(object):
    '''Computes and stores the sum, count and average'''
    def __init__(self):
        self.reset()

    def reset(self):    
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val 
        self.count += count

        if self.count==0:
            self.avg = 0
        else:
            self.avg = float(self.sum) / self.count

def str2bool(input):
    if isinstance(input, bool):
       return input
    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def Set_Param_Optim(args, model):
    """Set Parameters for optimization."""
    
    if isinstance(model, nn.DataParallel):
        return model.module.get_parameters()

    return model.get_parameters()

def Set_Optimizer(args, parameter_list, lr=0.001, weight_decay=0.0005, momentum=0.9):
    """Set Optimizer."""
    
    return optim.SGD(parameter_list, lr=lr, weight_decay=weight_decay, momentum=momentum)

def lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = lr * (1 + gamma * iter_num) ** (-power)

    for param_group in optimizer.param_groups:
        
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:    
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr

def lr_scheduler_withoutDecay(optimizer, lr=0.001, weight_decay=0.0005):
    """Learning rate without Decay."""

    for param_group in optimizer.param_groups:
        
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:    
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr

def Compute_Accuracy(args, pred, target, acc, prec, recall):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples.'''

    pred = pred.cpu().data.numpy()
    pred = np.argmax(pred, axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0],)
    target = target.astype(np.int32).reshape(target.shape[0],)

    for i in range(7):
        TP = np.sum((pred == i)*(target == i))
        TN = np.sum((pred != i)*(target != i))
        
        # Compute Accuracy of All --> TP+TN / All
        acc[i].update(np.sum(pred == target), pred.shape[0])
        
        # Compute Precision of Positive --> TP/(TP+FP)
        prec[i].update(TP, np.sum(pred==i))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[i].update(TP, np.sum(target==i))


def Count_Probility_Accuracy(six_probilities, six_accuracys, preds, target):
    preds = [np.argmax(pred.cpu().data.numpy(), axis=1) for pred in preds]
    target = target.cpu().data.numpy()
    
    #@ situation_0: global == global_local
    six_probilities[0].update(np.sum(preds[0] == preds[6]), len(preds[0]))
    six_accuracys[0].update(np.sum((preds[0] == preds[6]) * (preds[0] == target)), len(preds[0]))
    boolMatrics = (preds[0] == preds[6])

    #@ situation_1: global == global_local && one local also predict the same
    for img_index in range(len(preds[0])):
        cnt = 0
        if boolMatrics[img_index]: # 前提条件是global 和 global_local分类器预测值是相同的
            if preds[1][img_index] == preds[0][img_index]:
                cnt += 1
            if preds[2][img_index] == preds[0][img_index]:
                cnt += 1
            if preds[3][img_index] == preds[0][img_index]:
                cnt += 1
            if preds[4][img_index] == preds[0][img_index]:
                cnt += 1
            if preds[5][img_index] == preds[0][img_index]:
                cnt += 1
        for statisc in range(1, cnt + 1):
            six_probilities[statisc].update(1, 1) # 正更新
            if preds[0][img_index] == target[img_index]: # 预测正确
                six_accuracys[statisc].update(1, 1)
            else:
                six_accuracys[statisc].update(0, 1)

        for statisc_ in range(cnt+1, 6):
            six_probilities[statisc_].update(0, 1) # 负更新
            # six_accuracys[statisc_].update(0, 1)

def Count_Probility_Accuracy_Entropy(entropy_thresholds, probilities, accuracys,  preds_, target):
    '''
    entropy_thresholds: 信息熵的阈值列表，[0, log2(7)]，n等分，所以这个列表的长度是n
    probilities: 长度为n, probilities[i]代表的是在七个分类器都相同的情况下，且七个分类器的信息熵都小于entropy_thresholds[i]的概率
    accuracys: 长度为n, accuracys[i]代表的是在七个分类器都相同的情况下，且七个分类器的信息熵都小于entropy_thresholds[i]的情况下，七个分类器给出的共同的伪标签跟gt对比之下的可信度
    preds: shape是(7, batchSize, 7), 没有经过softmax处理的原始预测值
    target: shape是(batchSize), 每张图片的真是groundtruth
    '''

    preds = [np.argmax(pred.cpu().data.numpy(), axis=1) for pred in preds_]     # shape is (7, batchSize,)
    target = target.cpu().data.numpy()

    softmaxs = [nn.Softmax(dim=1)(preds_[i]) for i in range(7)]    # shape是（7, batchSize,）
    entropys = [torch.sum(-softmaxs[i] * torch.log(softmaxs[i] + 1e-5), dim=1) for i in range(7)] #计算七个分类器的信息熵, shape是(7, batchSize, 1)
    
    num_thresholds = len(entropy_thresholds)    
    boolMatrics = (preds[0] == preds[6]) # global 和 global + local 这两个分类器的预测值相同的0-1矩阵

    #@ 七个分类器预测值都相同且七个分类器的信息熵都小于各个阈值的情况的数量统计和给出的伪标签的可信度
    for enp_index in range(num_thresholds):
        for img_index in range(len(preds[0])):
            found = False
            if preds[0][img_index] == preds[1][img_index] == preds[2][img_index] == preds[3][img_index] == preds[4][img_index] == preds[5][img_index]: # 七个分类器的预测值相同
                if entropys[0][img_index] < entropy_thresholds[enp_index] and entropys[1][img_index] < entropy_thresholds[enp_index] and entropys[2][img_index] < entropy_thresholds[enp_index] and\
                    entropys[3][img_index] < entropy_thresholds[enp_index] and entropys[4][img_index] < entropy_thresholds[enp_index] and entropys[5][img_index] < entropy_thresholds[enp_index] and entropys[6][img_index] < entropy_thresholds[enp_index]: # 七个分类器的信息熵
                    probilities[enp_index].update(1, 1)
                    if preds[0][img_index] == target[img_index]:
                        accuracys[enp_index].update(1, 1)
                    else:
                        accuracys[enp_index].update(0, 1)
                    found = True
            if not found:
                probilities[enp_index].update(0, 1)

def Collect_Labeled_Image_Indexs(entropy_threshold, indexs, preds_, target):
    preds = [np.argmax(pred.cpu().data.numpy(), axis=1) for pred in preds_]     # shape is (7, batchSize,)
    target = target.cpu().data.numpy()

    softmaxs = [nn.Softmax(dim=1)(preds_[i]) for i in range(7)]    # shape是（7, batchSize,）
    entropys = [torch.sum(-softmaxs[i] * torch.log(softmaxs[i] + 1e-5), dim=1) for i in range(7)] #计算七个分类器的信息熵, shape是(7, batchSize, 1)
    
    #@ 七个分类器预测值都相同且七个分类器的信息熵都小于各个阈值的情况的数量统计和给出的伪标签的可信度
    collect_indexs, fake_labels, true_labels = [], [], []
    for img_index in range(len(preds[0])):
        if preds[0][img_index] == preds[1][img_index] == preds[2][img_index] == preds[3][img_index] == preds[4][img_index] == preds[5][img_index]: # 七个分类器的预测值相同
            if entropys[0][img_index] < entropy_threshold and entropys[1][img_index] < entropy_threshold and entropys[2][img_index] < entropy_threshold and \
                entropys[3][img_index] < entropy_threshold and entropys[4][img_index] < entropy_threshold and entropys[5][img_index] < entropy_threshold and entropys[6][img_index] < entropy_threshold: # 七个分类器的信息熵
                collect_indexs.append(indexs[img_index])
                fake_labels.append(preds[0][img_index])
                true_labels.append(target[img_index])
    return collect_indexs, fake_labels, true_labels

def BulidModel(args):
    """Bulid Model."""

    if args.useLocalFeature: # 选取主干
        if args.Backbone == 'ResNet18':
            model = IR(18, args.useIntraGCN, args.useInterGCN, args.useRandomMatrix, args.useAllOneMatrix, args.useCov, args.useCluster)
        elif args.Backbone == 'ResNet50':
            model = IR(50, args.useIntraGCN, args.useInterGCN, args.useRandomMatrix, args.useAllOneMatrix, args.useCov, args.useCluster)
        elif args.Backbone == 'VGGNet':
            model = VGG(args.useIntraGCN, args.useInterGCN, args.useRandomMatrix, args.useAllOneMatrix, args.useCov, args.useCluster)
        elif args.Backbone == 'MobileNet':
            model = MobileNetV2(args.useIntraGCN, args.useInterGCN, args.useRandomMatrix, args.useAllOneMatrix, args.useCov, args.useCluster)
    else:
        if args.Backbone == 'ResNet18':
            model = IR_onlyGlobal(18)
        elif args.Backbone == 'ResNet50':
            model = IR_onlyGlobal(50)
        elif args.Backbone == 'VGGNet':
            model = VGG_onlyGlobal()
        elif args.Backbone == 'MobileNet':
            model = MobileNetV2_onlyGlobal()

    if args.Resume_Model != 'None':
        print('Resume Model: {}'.format(args.Resume_Model))
        checkpoint = torch.load(args.Resume_Model, map_location='cuda:0')
        model.load_state_dict(checkpoint, strict=False)
    else:
        print('No Resume Model')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    return model

def BuildLabeledDataloader(args, dataloader, init_dataset_data, model, flag='train'):
    """Test."""
    model.eval()
    bar = tqdm(dataloader)
    data_imgs, data_labels, data_bboxs, data_landmarks, target_labels = [], [], [], [], []
    collect_num = 0
    for batch_index, (indexs, input, landmark, target) in enumerate(bar):
        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()

        # Forward Propagation
        with torch.no_grad():
            features, preds = model(input, landmark)

        collect_list_index, fake_labels,  true_labels = Collect_Labeled_Image_Indexs(0.19459, indexs, preds, target)
        data_labels += fake_labels
        target_labels += true_labels
        collect_num += len(collect_list_index)
        for c_index in collect_list_index:
            data_imgs.append(init_dataset_data['imgs_list'][c_index])
            data_bboxs.append(init_dataset_data['bboxs_list'][c_index])
            data_landmarks.append(init_dataset_data['landmarks_list'][c_index])
        bar.desc = f'[Collecting the fake labeled images from target domain]'
    trans = init_dataset_data['transform']
    probility = len(data_labels) / (len(dataloader) * args.train_batch_size)
    confidence = np.sum(np.array(data_labels) == np.array(target_labels)) / len(data_labels)
    print(f"collect {collect_num} images from target domain to train. probility is {probility}, confidence is {confidence}")

    
    trans = transforms.Compose([
            transforms.Resize((args.faceScale, args.faceScale)), # 112 * 112
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])


    # DataSet
    data_set = MyDataset(data_imgs, data_labels, data_bboxs, data_landmarks, flag, trans) # trans是tansformer处理，data_set是一个图片集，包括裁剪后的人脸图片，图片的五个关键点，表情标签
    if flag == 'train':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.train_batch_size, shuffle=True, num_workers=8, drop_last=True)
    elif flag == 'test':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.test_batch_size, shuffle=False, num_workers=8, drop_last=False)

    return data_loader, confidence, probility

def BulidAdversarialNetwork(args, model_output_num, class_num=7):
    """Bulid Adversarial Network."""

    if args.randomLayer:
        random_layer = RandomLayer([model_output_num, class_num], 1024)
        ad_net = AdversarialNetwork(1024, 512)

        random_layer.cuda()
        
    else:
        random_layer = None
        if args.methodOfDAN=='DANN':
            ad_net = AdversarialNetwork(model_output_num, 128)
        else:
            ad_net = AdversarialNetwork(model_output_num * class_num, 512)

    ad_net.cuda()

    return random_layer, ad_net

def BulidDataloader(args, flag1='train', flag2='source'):
    """Bulid data loader."""
    '''
    assert的用法:assert 条件,"报错信息"
    只有满足了条件，程序才会继续往下运行，否则就报自己设置的那个错误
    '''
    assert flag1 in ['train', 'test'], 'Function BuildDataloader : function parameter flag1 wrong.'
    assert flag2 in ['source', 'target'], 'Function BuildDataloader : function parameter flag2 wrong.'

    hug_change=transforms.ColorJitter(brightness=0.5)
    contrast_change=transforms.ColorJitter(brightness=0.5)
    brightness_change=transforms.ColorJitter(brightness=0.5)

    # Set Transform
    trans = transforms.Compose([
            transforms.Resize((args.faceScale, args.faceScale)), # 112 * 112
            hug_change=transforms.ColorJitter(brightness=0.5),
            contrast_change=transforms.ColorJitter(brightness=0.5),
            brightness_change=transforms.ColorJitter(brightness=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])c
    target_trans = None

    # Basic Notes:
    # 0: Surprised
    # 1: Fear
    # 2: Disgust
    # 3: Happy
    # 4: Sad
    # 5: Angry
    # 6: Neutral

    dataPath_prefix = '../Dataset'

    data_imgs, data_labels, data_bboxs, data_landmarks = [], [], [], []
    if flag1 == 'train':
        if flag2 == 'source':
            if args.sourceDataset == 'RAF': # RAF Train Set 用RAF作为源训练集
                list_patition_label = pd.read_csv(os.path.join(dataPath_prefix+'/RAF/basic/EmoLabel/list_patition_label.txt'), header=None, delim_whitespace=True) # delim_whitespace设置为True代表每一行以空格分开
                list_patition_label = np.array(list_patition_label)  # 在这之后的list_patition_label的shpae是15339*2，举个例子每一个单项就是'train_3066.jpg' 2 左边是图片的名字，右边是答案标签
                for index in range(list_patition_label.shape[0]):    # 遍历15339次
                    if list_patition_label[index, 0][:5] == "train": # list_patition_label[index,0][:5]的意思==list_patition_label[index][0][0:5]
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt'): # 去边框的文件夹中查找， 如果不存在就不用这张图片
                            continue
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt'): # ! 为什么ladmark_5的数据是五行数字而已，具体要怎么发挥这个的作用（我觉得有可能是五个点的意思）
                            continue
                        bbox = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int) # 把他转成整型,bbox bounding box的缩写
                        landmark = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int) # 注意了，list_patiotion_label的顺序是从训练集开始的，而bounding和Landmarks_5的txt文件都是从test开始排列的

                        data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0]) # 把图片存储路径用数组进行保存
                        data_labels.append(list_patition_label[index, 1]-1)  # 把图片对应的表情标签也按序号进行保存
                        data_bboxs.append(bbox)                              # 把每一张图片的人脸的轮廓的矩形的坐标点记录在这个列表里边，序号对应
                        data_landmarks.append(landmark) # 将一个人脸上的五个点的位置进行存储

            elif args.sourceDataset == 'AFED': # AFED Train Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 } #! ↓↓↓
                '''
                因为AFED这个数据集有11表情标签，并且标签的顺序是跟这个代码所规定的的表情表情是不完全一致的，所以，所以需要把train_list里边的错误的标签进行重映射。
                同时因为这个train_list里边并没有2，7，8，10号标枪数据，所以不用做这个映射。
                '''
                list_patition_label = pd.read_csv(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label) # list_patition_label的shape是(32757,6)，每一唯的表示的是（图片的名字，人脸左上角的坐标，人脸右下角的坐标，表情的标签）

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys(): # 如果list中存在么有这几种标签的数据说明不符合本代码的结构，直接跳过
                        continue 

                    bbox = list_patition_label[index,1:5].astype(np.int) # 提取除人脸的框框

                    landmark = np.loadtxt(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    '''
                    找到所遍历到的这幅图的五个局部特征点
                    '''
                    
                    data_imgs.append(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0]) # 存图片
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]]) # 存标签
                    data_bboxs.append(bbox)  # 存人脸框框
                    data_landmarks.append(landmark) # 存五个局部特征点

            elif args.sourceDataset == 'MMI': # MMI Dataset
                
                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
            
            elif args.sourceDataset=='FER2013': # FER2013 Train Set
                
                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            if args.useMultiDatasets == 'True':

                if args.targetDataset!='CK+': # CK+ Dataset

                    for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                        Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop',expression))
                        for imgFile in Dirs:
                            imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop',expression,imgFile)
                            img = Image.open(imgPath).convert('RGB')
                            ori_img_w, ori_img_h = img.size
                            
                            if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                                continue
                            landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                            
                            data_imgs.append(imgPath)
                            data_labels.append(index)
                            data_bboxs.append((0,0,ori_img_w,ori_img_h))
                            data_landmarks.append(landmark)

                    for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                        Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression))
                        for imgFile in Dirs:
                            imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression,imgFile)
                            img = Image.open(imgPath).convert('RGB')
                            ori_img_w, ori_img_h = img.size
                            
                            if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                                continue
                            landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                            
                            data_imgs.append(imgPath)
                            data_labels.append(index)
                            data_bboxs.append((0,0,ori_img_w,ori_img_h))
                            data_landmarks.append(landmark)

                if args.targetDataset!='JAFFE': # JAFFE Dataset

                    list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                    list_patition_label = np.array(list_patition_label)

                    for index in range(list_patition_label.shape[0]):

                        if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1])
                        data_bboxs.append(bbox) 
                        data_landmarks.append(landmark)

                if args.targetDataset!='MMI': # MMI Dataset

                    MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                    list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                    list_patition_label = np.array(list_patition_label)

                    for index in range(list_patition_label.shape[0]):

                        if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                        data_labels.append(MMItoLabel[list_patition_label[index,1]])
                        data_bboxs.append(bbox) 
                        data_landmarks.append(landmark)

                if args.targetDataset!='Oulu-CASIA': # Oulu-CASIA Dataset

                    list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                    list_patition_label = np.array(list_patition_label)

                    for index in range(list_patition_label.shape[0]):

                        if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                            continue
                        
                        img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                        ori_img_w, ori_img_h = img.size

                        landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1])
                        data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                        data_landmarks.append(landmark)

        elif flag2 == 'target':
            if args.targetDataset == 'CK+': # CK+ Train Set
                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop', expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop', expression, imgFile)
                        img = Image.open(imgPath).convert('RGB')
                        ori_img_w, ori_img_h = img.size # 因为已经裁剪好了，但是为了统一使用bbox参数，所以就用了原图size的尺寸
                        
                        if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/', expression,imgFile[:-3]+'txt')):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/', expression,imgFile[:-3]+'txt')).astype(np.int)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append((0, 0, ori_img_w, ori_img_h))
                        data_landmarks.append(landmark)

            elif args.targetDataset=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1]) # 存标签
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='MMI': # MMI Dataset

                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='SFEW': # SFEW Train Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Bboxs/',expression))
                    for bboxName in Dirs:
                        bboxsPath = os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Bboxs/',expression,bboxName)
                        bboxs = np.loadtxt(bboxsPath).astype(np.int)

                        if not os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Landmarks_5/',expression,bboxName)):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Landmarks_5/',expression,bboxName)).astype(np.int)

                        if os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'png')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'png')
                        elif os.path.exists(os.path.join(dataPath_prefix+'/SFEW//Train/imgs/',expression,bboxName[:-3]+'jpg')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'jpg')
                        else:
                            print(os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'*') + ' no exist')

                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append(bboxs)
                        data_landmarks.append(landmark)

            elif args.targetDataset=='FER2013': # FER2013 Train Set
                
                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            elif args.targetDataset=='ExpW': # ExpW Train Set
                
                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/train_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
            
            elif args.targetDataset=='AFED': # AFED Train Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)  
                    data_landmarks.append(landmark)

            elif args.targetDataset=='WFED': # WFED Train Set

                WesternToLabel = { 2:0, 5:1, 4:2, 1:3, 3:4, 6:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Western_Films_Expression_Datasets/list/train_random.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    
                    if not os.path.exists(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt').astype(np.int)
                    
                    imgPath = dataPath_prefix+'/Western_Films_Expression_Datasets/images/'+list_patition_label[index,0]
                    if os.path.exists(imgPath+'.png'):
                        data_imgs.append(imgPath+'.png')
                    elif os.path.exists(imgPath+'.jpg'):
                        data_imgs.append(imgPath+'.jpg')
                    else:
                        continue

                    data_labels.append(WesternToLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.targetDataset == 'RAF': # RAF Train Set

                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,0][:5] == "train":

                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1]-1)
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)

    elif flag1 == 'test':
        if flag2 =='source':
            if args.sourceDataset=='CK+': # CK+ Val Set
                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression,imgFile)
                        img = Image.open(imgPath).convert('RGB')
                        ori_img_w, ori_img_h = img.size
                        
                        if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append((0,0,ori_img_w,ori_img_h))
                        data_landmarks.append(landmark)

            elif args.sourceDataset=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='MMI': # MMI Dataset

                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='SFEW': # SFEW 2.0 Val Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression))
                    for bboxName in Dirs:
                        bboxsPath = os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression,bboxName)
                        bboxs = np.loadtxt(bboxsPath).astype(np.int)

                        if not os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)).astype(np.int)

                        if os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')
                        elif os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')
                        else:
                            print(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'*') + ' no exist')

                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append(bboxs)
                        data_landmarks.append(landmark)

            elif args.sourceDataset=='FER2013': # FER2013 Val Set

                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='ExpW': # ExpW Val Set

                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/val_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
           
            elif args.sourceDataset=='AFED': # AFED Val Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='WFED': # WFED Val Set

                WesternToLabel = { 2:0, 5:1, 4:2, 1:3, 3:4, 6:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Western_Films_Expression_Datasets/list/val_random.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    
                    if not os.path.exists(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt').astype(np.int)
                    
                    imgPath = dataPath_prefix+'/Western_Films_Expression_Datasets/images/'+list_patition_label[index,0]
                    if os.path.exists(imgPath+'.png'):
                        data_imgs.append(imgPath+'.png')
                    elif os.path.exists(imgPath+'.jpg'):
                        data_imgs.append(imgPath+'.jpg')
                    else:
                        continue

                    data_labels.append(WesternToLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='RAF': # RAF Test Set
                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index, 0][:4] == "test":

                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index, 0])
                        data_labels.append(list_patition_label[index, 1] - 1)
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)

        elif flag2 == 'target':
            if args.targetDataset == 'CK+': # CK+ Val Set

                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop', expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop', expression, imgFile)
                        img = Image.open(imgPath).convert('RGB')
                        ori_img_w, ori_img_h = img.size
                        
                        if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append((0, 0, ori_img_w, ori_img_h))
                        data_landmarks.append(landmark)

            elif args.targetDataset=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index, 1])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.targetDataset=='MMI': # MMI Dataset

                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='SFEW': # SFEW 2.0 Val Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression))
                    for bboxName in Dirs:
                        bboxsPath = os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression,bboxName)
                        bboxs = np.loadtxt(bboxsPath).astype(np.int)

                        if not os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)).astype(np.int)

                        if os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')
                        elif os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')
                        else:
                            print(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'*') + ' no exist')

                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append(bboxs)
                        data_landmarks.append(landmark)

            elif args.targetDataset=='FER2013': # FER2013 Val Set

                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            elif args.targetDataset=='ExpW': # ExpW Val Set

                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/val_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
           
            elif args.targetDataset=='AFED': # AFED Val Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.targetDataset=='WFED': # WFED Val Set

                WesternToLabel = { 2:0, 5:1, 4:2, 1:3, 3:4, 6:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Western_Films_Expression_Datasets/list/val_random.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    
                    if not os.path.exists(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt').astype(np.int)
                    
                    imgPath = dataPath_prefix+'/Western_Films_Expression_Datasets/images/'+list_patition_label[index,0]
                    if os.path.exists(imgPath+'.png'):
                        data_imgs.append(imgPath+'.png')
                    elif os.path.exists(imgPath+'.jpg'):
                        data_imgs.append(imgPath+'.jpg')
                    else:
                        continue

                    data_labels.append(WesternToLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.targetDataset=='RAF': # RAF Test Set

                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,0][:4] == "test":

                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1]-1)
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)

    # DataSet Distribute
    distribute_ = np.array(data_labels) # 存图片的标签的
    print('The %s %s dataset quantity: %d' % (flag1, flag2, len(data_imgs)))
    print('The %s %s dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (flag1, flag2,
           np.sum(distribute_ == 0), np.sum(distribute_==1), np.sum(distribute_==2), np.sum(distribute_==3),
           np.sum(distribute_ == 4), np.sum(distribute_==5), np.sum(distribute_==6))) # 这里整个数据集中，7种类别的表情的数量

    # DataSet
    data_set = MyDataset(data_imgs, data_labels, data_bboxs, data_landmarks, flag1, trans, target_trans) # trans是tansformer处理，data_set是一个图片集，包括裁剪后的人脸图片，图片的五个关键点，表情标签
    # DataLoader
    if flag1 == 'train':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.train_batch_size, shuffle=True, num_workers=8, drop_last=True)
    elif flag1 == 'test':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.test_batch_size, shuffle=False, num_workers=8, drop_last=False)

    return data_loader, {'imgs_list':data_imgs, 'labels_list':data_labels, 'bboxs_list':data_bboxs, 'landmarks_list':data_landmarks, 'transform':trans}

def Show_Accuracy(acc, prec, recall, class_num=7):
    """Compute average of accuaracy/precision/recall/f1"""

    # Compute F1 value    
    f1 = [AverageMeter() for i in range(class_num)]
    for i in range(class_num):
        if prec[i].avg == 0 or recall[i].avg == 0:
            f1[i].avg = 0
            continue
        f1[i].avg = 2*prec[i].avg*recall[i].avg/(prec[i].avg+recall[i].avg)
    
    # Compute average of accuaracy/precision/recall/f1
    acc_avg, prec_avg, recall_avg, f1_avg = 0, 0, 0, 0

    for i in range(class_num):
        acc_avg+=acc[i].avg
        prec_avg+=prec[i].avg
        recall_avg+=recall[i].avg
        f1_avg+=f1[i].avg

    acc_avg, prec_avg, recall_avg, f1_avg = acc_avg/class_num, prec_avg/class_num, recall_avg/class_num, f1_avg/class_num

    # Log Accuracy Infomation
    Accuracy_Info = ''
    
    Accuracy_Info+='    Accuracy'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(acc[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Precision'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(prec[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Recall'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(recall[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    F1'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(f1[i].avg)
    Accuracy_Info+='\n'

    return Accuracy_Info, acc_avg, prec_avg, recall_avg, f1_avg

def Show_OnlyAccuracy(accuracys):
    accs = [0 for i in range(len(accuracys))] # 预设7个分类器的分类准确率为0
    for classifier_id in range(len(accuracys)):
        for class_id in range(len(accuracys[0])):
            accs[classifier_id] += accuracys[classifier_id][class_id].avg
    return [acc / len(accuracys) for acc in accs]

def Initialize_Mean(args, model, useClassify=True):
    
    model.eval()
    
    source_data_loader = BulidDataloader(args, flag1='train', flag2='source')
    target_data_loader = BulidDataloader(args, flag1='train', flag2='target')
    
    # Source Mean
    mean = None

    for step, (input, landmark, label) in enumerate(source_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Source')

        if step == 0:
            mean = torch.mean(feature,0)
        else:
            mean = step/(step+1) * torch.mean(feature,0) + 1/(step+1) * mean

    if isinstance(model, nn.DataParallel):
        model.module.SourceMean.init(mean)
    else:
        model.SourceMean.init(mean)

    # Target Mean
    mean = None

    for step, (input, landmark, label) in enumerate(target_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Target')

        if step==0:
            mean = torch.mean(feature,0)
        else:
            mean = step/(step+1) * torch.mean(feature,0) + 1/(step+1) * mean

    if isinstance(model, nn.DataParallel):
        model.module.TargetMean.init(mean)
    else:
        model.TargetMean.init(mean)

def Initialize_Mean_Cov(args, model, useClassify=True):

    model.eval()

    source_data_loader = BulidDataloader(args, flag1='train', flag2='source')
    target_data_loader = BulidDataloader(args, flag1='train', flag2='target')

    # Source Mean and Cov
    mean, cov = None, None

    for step, (input, landmark, label) in enumerate(source_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Source')

        if step == 0:
            mean = torch.mean(feature, 0)
            cov = torch.mm((feature-mean).transpose(0, 1), feature-mean) / (feature.size(0)-1) # cov是协方差
        else:
            mean = step/(step+1) * torch.mean(feature, 0) + 1/(step+1) * mean
            cov = step/(step+1) * torch.mm((feature-mean).transpose(0, 1), feature-mean) / (feature.size(0)-1) + 1/(step+1) * cov

    if isinstance(model, nn.DataParallel):
        model.module.SourceMean.init(mean, cov)
    else:
        model.SourceMean.init(mean, cov)

    # Target Mean and Cov
    mean, cov = None, None

    for step, (input, landmark, label) in enumerate(target_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Target')

        if step == 0:
            mean = torch.mean(feature, 0)
            cov = torch.mm((feature-mean).transpose(0, 1), feature-mean) / (feature.size(0)-1)
        else:
            mean = step/(step+1) * torch.mean(feature, 0) + 1/(step+1) * mean # 这个是怎么算出来的？
            cov = step/(step+1) * torch.mm((feature-mean).transpose(0, 1), feature-mean) / (feature.size(0)-1) + 1/(step+1) * cov

    if isinstance(model, nn.DataParallel):
        model.module.TargetMean.init(mean, cov)
    else:
        model.TargetMean.init(mean, cov)

def Initialize_Mean_Cluster(args, model, useClassify=True,source_data_loader=None,target_data_loader=None):

    model.eval() # 这会让model.training = False
    
    # Source Cluster of Mean
    Feature = []
    EndTime = time.time()
    # source_data_loader = BulidDataloader(args, flag1='train', flag2='source')

    source_data_bar = tqdm(source_data_loader)
    for step, (_, input, landmark, label) in enumerate(source_data_bar):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            #! m21-11-12，注释
            # feature, output, loc_output = model(input, landmark)
            #@ m21-11-11, 新的model调用
            features, preds = model(input, landmark)
        source_data_bar.desc = 'initialize the source domain'

        Feature.append(features[6].cpu().data.numpy())
    Feature = np.vstack(Feature)    # shape is (12256, 384) while source dataset is RAF-DB

    # Using K-Means
    kmeans = KMeans(n_clusters=args.class_num, init='k-means++', algorithm='full')
    kmeans.fit(Feature)
    centers = torch.Tensor(kmeans.cluster_centers_).to('cuda' if torch.cuda.is_available else 'cpu') # shape is (7, 384)，通过feature获取七个类别的分布点

    if isinstance(model, nn.DataParallel):
        model.module.SourceMean.init(centers)
    else:
        model.SourceMean.init(centers) # 通过这里初始化了SourceMean里边的self.running_mean

    print('[Source Domain] Cost time : %fs' % (time.time()-EndTime))

    # Target Cluster of Mean
    Feature = []
    EndTime = time.time()
    # target_data_loader = BulidDataloader(args, flag1='train', flag2='target')

    target_data_bar = tqdm(target_data_loader)
    for step, (_, input, landmark, label) in enumerate(target_data_bar):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            #! m21-11-12，注释
            # feature, output, loc_output = model(input, landmark)
            #@ m21-11-11, 新的model调用
            features, preds = model(input, landmark)
        target_data_bar.desc = 'initialize the target domain'

        Feature.append(features[6].cpu().data.numpy())
    Feature = np.vstack(Feature)

    # Using K-Means
    kmeans = KMeans(n_clusters=args.class_num, init='k-means++', algorithm='full')
    kmeans.fit(Feature)
    centers = torch.Tensor(kmeans.cluster_centers_).to('cuda' if torch.cuda.is_available else 'cpu')

    if isinstance(model, nn.DataParallel):
        model.module.TargetMean.init(centers)
    else:
        model.TargetMean.init(centers)

    print('[Target Domain] Cost time : %fs' % (time.time()-EndTime))

def Visualization(figName, model, dataloader=None, useClassify=True, domain='Source'):
    '''Feature Visualization in Source/Target Domain.'''
    
    assert useClassify in [True, False], 'useClassify should be bool.'
    assert domain in ['Source', 'Target'], 'domain should be source or target.'

    dataloader = tqdm(dataloader)
    model.eval() # 先设置不保存梯度

    Feature, Label = [], []

    # Get Cluster
    for i in range(7):
        if domain == 'Source':
            Feature.append(model.SourceMean.running_mean[i].cpu().data.numpy()) # 把七个聚类点先存进Features列表中
        elif domain == 'Target':
            Feature.append(model.TargetMean.running_mean[i].cpu().data.numpy())
    Label.append(np.array([7 for i in range(7)])) # 先预设7个聚类点，这7个点对应Feature里边的前7个点，作为聚类点

    # Get Feature and Label
    for step, (input, landmark, label) in enumerate(dataloader):
        '''
        参数解释
        input: 裁剪后的输入的图片，shape应该是128*128*3
        landmark: 五个关键点的坐标
        label: 该图片所属的表情标签
        '''
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain)
            '''
            feature: batch * 384
            output : batch * 7 全局的预测值
            loc_output: batch * 7 局部的预测值
            '''
        Feature.append(feature.cpu().data.numpy())
        Label.append(label.cpu().data.numpy())


    Feature = np.vstack(Feature) # (7+N, 384) where N is the number of datasets
    '''
    在执行这句代码之前，Feature的前7项每一项的shape都是1*384，这七项存的是预设的七个聚类点
    然后在用dataloader弹出的图片的循环之后Feature增添的每一项的shape都是batch*384，
    然后因为没有对vstack的axis参数赋值，所以默认是第0维，因此在执行这句代码之后，会将Feature
    列表中的每个元素的第0维进行叠加，最后就形成了一个(32771,384)的列表，其中每一行就代表这张图片的features特征
    '''
    Label = np.concatenate(Label) # (N+7, 1)

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature) # 对Features进行压缩，压缩后的shape会变成（32771，2）

    # Draw Visualization of Feature
    colors = {0:'red', 1:'blue', 2:'olive',  3:'green',  4:'orange',  5:'purple',  6:'darkslategray', 7:'black'}
    labels = {0:'Surprised', 1:'Fear', 2:'Disgust',  3:'Happy',  4:'Sad',  5:'Angry',  6:'Neutral', 7:'Cluster'}
    # labels = {0:'惊讶', 1:'恐惧', 2:'厌恶', 3:'开心', 4:'悲伤', 5:'愤怒', 6:'平静', 7:'聚类中心'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min) # 求均值，data_norm的shape是（32771，2）

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):
        data_x, data_y = data_norm[Label==i][:,0], data_norm[Label==i][:,1] # 找出所有label=i的点
        scatter = plt.scatter(data_x, data_y, c='m', edgecolors=colors[i], s=5, label=labels[i], marker='^', alpha=0.6) # 画图
    scatter = plt.scatter(data_norm[Label==7][:,0], data_norm[Label==7][:,1], c=colors[7], s=20, label=labels[7], marker='^', alpha=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    
    plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in range(8)],
               loc='upper left',
            #    prop = {'size':8},
               bbox_to_anchor=(1.05,0.85),
               borderaxespad=0)
    plt.savefig(fname='{}'.format(figName), format="jpg", bbox_inches = 'tight')
    # plt.show()

def VisualizationForTwoDomain(figName, model, source_dataloader, target_dataloader, useClassify=True, showClusterCenter=True):
    '''Feature Visualization in Source and Target Domain.'''
    
    model.eval()

    Feature_Source, Label_Source, Feature_Target, Label_Target = [], [], [], []

    # Get Feature and Label in Source Domain
    if showClusterCenter:
        for i in range(7):
            Feature_Source.append(model.SourceMean.running_mean[i].cpu().data.numpy())
        Label_Source.append(np.array([7 for i in range(7)]))   

    for step, (input, landmark, label) in enumerate(source_dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain='Source')

        Feature_Source.append(feature.cpu().data.numpy())
        Label_Source.append(label.cpu().data.numpy())

    Feature_Source = np.vstack(Feature_Source)
    Label_Source = np.concatenate(Label_Source)

    # Get Feature and Label in Target Domain
    if showClusterCenter:
        for i in range(7):
            Feature_Target.append(model.TargetMean.running_mean[i].cpu().data.numpy())
        Label_Target.append(np.array([7 for i in range(7)]))

    for step, (input, landmark, label) in enumerate(target_dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain='Target')

        Feature_Target.append(feature.cpu().data.numpy())
        Label_Target.append(label.cpu().data.numpy())

    Feature_Target = np.vstack(Feature_Target)
    Label_Target = np.concatenate(Label_Target)

    # Sampling from Source Domain
    Feature_Temple, Label_Temple = [], []
    for i in range(8):
        num_source = np.sum(Label_Source == i)
        num_target = np.sum(Label_Target == i)

        num = num_source if num_source <= num_target else num_target 

        Feature_Temple.append(Feature_Source[Label_Source == i][:num])
        Label_Temple.append(Label_Source[Label_Source == i][:num])
 
    Feature_Source = np.vstack(Feature_Temple)
    Label_Source = np.concatenate(Label_Temple)

    Label_Target += 8 # 加8是为了后面直接通过Label == i 来获取target中的标签，因为Source和Target已经合并在一起了，通过这个方法就很巧

    Feature = np.vstack((Feature_Source, Feature_Target))
    Label = np.concatenate((Label_Source, Label_Target)) # 做成了一个元祖

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0:'firebrick', 1:'aquamarine', 2:'goldenrod',  3:'cadetblue',  4:'saddlebrown',  5:'yellowgreen',  6:'navy'}
    labels = {0:'Surprised', 1:'Fear', 2:'Disgust',  3:'Happy',  4:'Sad',  5:'Angry',  6:'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):

        data_source_x, data_source_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
        source_scatter = plt.scatter(data_source_x, data_source_y, color="none", edgecolor=colors[i], s=20, label=labels[i], marker="o", alpha=0.4, linewidth=0.5)
        
        data_target_x, data_target_y = data_norm[Label == (i+8)][:, 0], data_norm[Label == (i+8)][:, 1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor="none", s=30, label=labels[i], marker="x", alpha=0.6, linewidth=0.2)

        if i == 0:
            source_legend = source_scatter
            target_legend = target_scatter

    if showClusterCenter:
        source_cluster = plt.scatter(data_norm[Label == 7][:, 0], data_norm[Label == 7][:, 1], c='black', s=20, label='Source Cluster Center', marker='^', alpha=1)
        target_cluster = plt.scatter(data_norm[Label == 15][:, 0], data_norm[Label == 15][:, 1], c='black', s=20, label='Target Cluster Center', marker='s', alpha=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    '''
    l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], 
                    label="{:s}".format(labels[i]) ) for i in range(7)], 
                    loc='upper left', 
                    prop = {'size':8})
                    #bbox_to_anchor=(1.05,0.85), 
                    #borderaxespad=0)
    
    if showClusterCenter:
        plt.legend([source_legend, target_legend, source_cluster, target_cluster],
                   ['Source Domain', 'Target Domain', 'Source Cluster Center', 'Target Cluster Center'],
                   loc='lower left',
                   prop = {'size':7})
    else:
        plt.legend([source_legend, target_legend], ['Source Domain', 'Target Domain'], loc='lower left', prop = {'size':7})
    plt.gca().add_artist(l1)
    '''
    plt.savefig(fname='{}.jpg'.format(figName), format="jpg", bbox_inches='tight')

def makeFolder(args):
    os.makedirs(args.OutputPath)
    os.makedirs(args.OutputPath + '/result_pics/train/source')
    os.makedirs(args.OutputPath + '/result_pics/train/target')
    os.makedirs(args.OutputPath + '/result_pics/test/source')
    os.makedirs(args.OutputPath + '/result_pics/test/target')
    os.makedirs(args.OutputPath + '/result_pics/train_tow_domain')
    os.makedirs(args.OutputPath + '/result_pics/test_tow_domain')
    os.mkdir(args.OutputPath + '/code')
    prefixPath = args.OutputPath + '/code'
    if not os.path.exists(prefixPath):
        os.mkdir(prefixPath)
    for fileName in ['AdversarialNetwork.py', 'Dataset.py', 'demo.py', 'getPreTrainedModel_ResNet.py', 'GraphConvolutionNetwork.py', 'Loss.py', 'Model.py',
        'ResNet.py', 'TrainOnSourceDomain.py', 'TrainOnSourceDomain.sh', 'TransferToTargetDomain.py', 'TransferToTargetDomain.sh', 'Utils.py']:
        try:
            shutil.copyfile(fileName, os.path.join(prefixPath, fileName.split('/')[-1]))
        except:
            continue