import os
import sys
import time
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from Loss import HAFN, SAFN
from Utils import *

def construct_args():
    parser = argparse.ArgumentParser(description='Expression Classification Training')

    parser.add_argument('--Log_Name', type=str,default='ResNet50_CropNet_GCNwithIntraMatrixAndInterMatrix_useCluster_withoutAFN_trainOnSourceDomain_RAFtoAFED', help='Log Name')
    parser.add_argument('--OutputPath', type=str,default='.', help='Output Path')
    parser.add_argument('--Backbone', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet']) # 挑选backbone
    parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='None') # 导入pretrained模型用的
    parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')  # 选择的gpu型号

    parser.add_argument('--useAFN', type=str2bool, default=False, help='whether to use AFN Loss')
    parser.add_argument('--methodOfAFN', type=str, default='SAFN', choices=['HAFN', 'SAFN']) #  AFN --  Adaptive Feature Norm 一种特征的自适应方法
    parser.add_argument('--radius', type=float, default=40, help='radius of HAFN (default: 25.0)') # k-means计算的半径
    parser.add_argument('--deltaRadius', type=float, default=0.001, help='radius of SAFN (default: 1.0)') # ! 这个跟上面那个半径有什么区别
    parser.add_argument('--weight_L2norm', type=float, default=0.05, help='weight L2 norm of AFN (default: 0.05)') # AFN是一种求损失的方法

    # ! dataset
    parser.add_argument('--faceScale', type=int, default=112, help='Scale of face (default: 112)') # 人脸图片的尺寸
    parser.add_argument('--sourceDataset', type=str, default='AFED', choices=['RAF', 'AFED', 'MMI', 'FER2013']) # source dataset的名字
    parser.add_argument('--targetDataset', type=str, default='JAFFE', choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED']) # 目标域
    parser.add_argument('--train_batch_size', type=int, default=32, help='input batch size for training (default: 64)') # 训练的batch size
    parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size for testing (default: 64)')   # 测试集的batch size
    parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')         #  是否使用多个数据集（使用多个数据集的什么意思呢）

    parser.add_argument('--lr', type=float, default=0.0001) # 学习率
    parser.add_argument('--epochs', type=int, default=60,help='number of epochs to train (default: 10)') # 训练代数
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')       # 动量 
    parser.add_argument('--weight_decay', type=float, default=0.0001,help='SGD weight decay (default: 0.0005)') # 正则项系数

    parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model') # 是否只是进行测试，意思就是不用训练
    parser.add_argument('--showFeature', type=str2bool, default=True, help='whether to show feature') # 展示特征（咩啊，用来干嘛的，展示特征图把）

    parser.add_argument('--useIntraGCN', type=str2bool, default=True, help='whether to use Intra-GCN') #  在域内是否使用GCN传播
    parser.add_argument('--useInterGCN', type=str2bool, default=True, help='whether to use Inter-GCN') #  在域间是否使用GCN传播
    parser.add_argument('--useLocalFeature', type=str2bool, default=True, help='whether to use Local Feature') # 是否使用局部特征

    parser.add_argument('--useRandomMatrix', type=str2bool, default=False, help='whether to use Random Matrix') # 这个是用来初始化GCN构造的那个矩阵的一种方法
    parser.add_argument('--useAllOneMatrix', type=str2bool, default=False, help='whether to use All One Matrix') # 这个也是用来初始化GCN构造的那个矩阵的一种方法

    parser.add_argument('--useCov', type=str2bool, default=False, help='whether to use Cov') #  一种画图的方法
    parser.add_argument('--useCluster', type=str2bool, default=True, help='whether to use Cluster') # 后面在话TSNE画图的时候用到的

    parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)') #  最后输出的分类的类别数目 
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')          #  随机种子

    return parser.parse_args()


def Train(args, model, train_dataloader, optimizer, epoch, writer):
    """Train."""

    model.train()
    # torch.autograd.set_detect_anomaly(True) # 正向传播的时候开启求导异常的检测

    #! 注释
    # acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)] # 每个变量都装载了7个AverageMeter对象，AverageMeter对象存储变量易于更新和修改。
    # loss, global_cls_loss, local_cls_loss, afn_loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    
    acc, prec, recall = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    cls_loss, loss, afn_loss, data_time, batch_time = [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities, six_accuracys = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]

    # Decay Learn Rate per Epoch，不同的主干采用不同的学习率迭代
    if args.Backbone in ['ResNet18', 'ResNet50']:
        if epoch <= 10:
            args.lr = 1e-4
        elif epoch <= 40:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.Backbone == 'MobileNet':
        if epoch <= 20:
            args.lr = 1e-3
        elif epoch <= 40:
            args.lr = 1e-4
        elif epoch <= 60:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.Backbone == 'VGGNet':
        if epoch <= 30:
            args.lr = 1e-3
        elif epoch <= 60:
            args.lr = 1e-4
        elif epoch <= 70:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    end = time.time()
    train_bar = tqdm(train_dataloader)
    for step, (_, input, landmark, label) in enumerate(train_bar):
        '''
            input: (3, batch, 112, 112)
            landmark: (batch, 5, 2)
            label: (batch, 1)
        '''
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        data_time.update(time.time() - end)

        # Forward propagation
        end = time.time()
        #! m21-11-11, 注释
        # feature, output, loc_output = model(input, landmark)
        #@ m21-11-11, 新的model调用
        features, preds = model(input, landmark)
        batch_time.update(time.time() - end)

        # Compute Loss
        #! m21-11-11, 注释
        '''
            global_cls_loss_ = nn.CrossEntropyLoss()(output, label)
            local_cls_loss_ = nn.CrossEntropyLoss()(loc_output, label) if args.useLocalFeature else 0
            afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN == 'HAFN' else SAFN(feature, args.weight_L2norm, args.deltaRadius)) if args.useAFN else 0
            loss_ = global_cls_loss_ + local_cls_loss_ + (afn_loss_ if args.useAFN else 0)
        '''

        #@ m21-11-11, 新的计算loss的方式
        loss_ = 0
        classifiers_loss_ratio = [5, 1, 1, 1, 1, 1, 5]
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i], label)
            cls_loss[i].update(tmp.cpu().data.item(), input.size(0))
            loss_ += classifiers_loss_ratio[i] * tmp
        
        # Back Propagation
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss_.backward()
        optimizer.step()

        # Decay Learn Rate
        optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr, weight_decay=args.weight_decay) # optimizer = lr_scheduler(optimizer, num_iter*(epoch-1)+step, 0.001, 0.75, lr=args.lr, weight_decay=args.weight_decay)
        # Compute accuracy, recall and loss
        for classifier_id in range(7): # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id], label, acc[classifier_id], prec[classifier_id], recall[classifier_id])

        Count_Probility_Accuracy(six_probilities, six_accuracys, preds, label)

        # Log loss
        loss.update(float(loss_.cpu().data.item()), input.size(0))

        end = time.time()
        train_bar.desc = f'[Train (Source Domain) Epoch {epoch}/{args.epochs}] cls loss: {cls_loss[0].avg:.3f}, {cls_loss[1].avg:.3f}, {cls_loss[2].avg:.3f}, {cls_loss[3].avg:.3f}, {cls_loss[4].avg:.3f}, {cls_loss[5].avg:.3f}, {cls_loss[6].avg:.3f}'
    
    #! m21-11-12, 注释
    # AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    #@ m21-11-12, 只记录七个分类器的准确率
    accs = Show_OnlyAccuracy(acc)
    writer.add_scalars('Accuracy/Train', {'global': accs[0], 'left_eye':accs[1], 'right_eye':accs[2], 'nose':accs[3], 'left_mouse':accs[4], 'right_mouse':accs[5], 'global_local':accs[6]}, epoch)
    writer.add_scalars('Loss/Train', {'global': cls_loss[0].avg, 'left_eye':cls_loss[1].avg, 'right_eye':cls_loss[2].avg, 'nose':cls_loss[3].avg, 'left_mouse':cls_loss[4].avg, 'right_mouse':cls_loss[5].avg, 'global_local':cls_loss[6].avg}, epoch)
    writer.add_scalars('Six_Probilities/Train', {'situation_0': six_probilities[0].avg, 'situation_1':six_probilities[1].avg, 'situation_2':six_probilities[2].avg, 'situation_3':six_probilities[3].avg, 'situation_4':six_probilities[4].avg, 'situation_5':six_probilities[5].avg}, epoch)
    writer.add_scalars('Six_Accuracys/Train', {'situation_0': six_accuracys[0].avg, 'situation_1':six_accuracys[1].avg, 'situation_2':six_accuracys[2].avg, 'situation_3':six_accuracys[3].avg, 'situation_4':six_accuracys[4].avg, 'situation_5':six_accuracys[5].avg}, epoch)

    LoggerInfo = '[Train (Source Domain) Epoch {0}]： Learning Rate {1} \n'\
    'Accuracy: global {Accuracys[0]:.4f}\t left_eye {Accuracys[1]:.4f}\t right_eye {Accuracys[2]:.4f}\t nose {Accuracys[3]:.4f}\t left_mouse {Accuracys[4]:.4f}\t right_mouse {Accuracys[5]:.4f}\t global_local {Accuracys[6]:.4f}\n'\
    'Cls_Loss: global {Loss[0].avg:.4f}\t left_eye {Loss[1].avg:.4f}\t right_eye {Loss[2].avg:.4f}\t nose {Loss[3].avg:.4f}\t left_mouse {Loss[4].avg:.4f}\t right_mouse {Loss[5].avg:.4f}\t global_local {Loss[6].avg:.4f}\n'\
    'Situ_Acc: Situation_0 {six_acc[0].avg:.4f}\t Situation_1 {six_acc[1].avg:.4f}\t Situation_2 {six_acc[2].avg:.4f}\t Situation_3 {six_acc[3].avg:.4f}\t Situation_4 {six_acc[4].avg:.4f}\t Situation_5 {six_acc[5].avg:.4f}\n'\
    'Situ_Pro: Situation_0 {six_prob[0].avg:.4f}\t Situation_1 {six_prob[1].avg:.4f}\t Situation_2 {six_prob[2].avg:.4f}\t Situation_3 {six_prob[3].avg:.4f}\t Situation_4 {six_prob[4].avg:.4f}\t Situation_5 {six_prob[5].avg:.4f}\n\n'\
    .format(epoch, args.lr, Accuracys=accs, Loss=cls_loss, six_acc=six_accuracys, six_prob=six_probilities)

    with open(args.OutputPath + "/train_result.log", "a") as f:
        f.writelines(LoggerInfo)

def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Acc, epoch, confidence_target, writer):
    """Test."""

    model.eval()
    # torch.autograd.set_detect_anomaly(True)

    #! m21-11-12
    # iter_source_dataloader = iter(test_source_dataloader)
    # iter_target_dataloader = iter(test_target_dataloader)

    # Test on Source Domain
    acc, prec, recall = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    cls_loss, loss, afn_loss, data_time, batch_time = [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities, six_accuracys = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]
    
    end = time.time()
    test_source_bar = tqdm(test_source_dataloader)
    for step, (_, input, landmark, label) in enumerate(test_source_bar):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        data_time.update(time.time() - end)
        
        # Forward Propagation
        with torch.no_grad():
            end = time.time()
            #! m21-11-12，注释
            # feature, output, loc_output = model(input, landmark)
            #@ m21-11-11, 新的model调用
            features, preds = model(input, landmark)
            batch_time.update(time.time()-end)
        
        # Compute Loss
        #! m21-11-11, 注释                                                                                                                                                                                                                                                                                                                                                                                                      
        '''
            global_cls_loss_ = nn.CrossEntropyLoss()(output, label)
            local_cls_loss_ = nn.CrossEntropyLoss()(loc_output, label) if args.useLocalFeature else 0
            afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN == 'HAFN' else SAFN(feature, args.weight_L2norm, args.deltaRadius)) if args.useAFN else 0
            loss_ = global_cls_loss_ + local_cls_loss_ + (afn_loss_ if args.useAFN else 0)
        '''

        #@ m21-11-11, 新的计算loss的方式
        loss_ = 0
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i], label)
            cls_loss[i].update(tmp.cpu().data.item(), input.size(0))
            loss_ += tmp

        # Compute accuracy, precision and recall
        for classifier_id in range(7): # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id], label, acc[classifier_id], prec[classifier_id], recall[classifier_id])

        Count_Probility_Accuracy(six_probilities, six_accuracys, preds, label)

        # Log loss
        loss.update(float(loss_.cpu().data.item()), input.size(0))
        end = time.time()
        test_source_bar.desc = f'[Test (Source Domain) Epoch {epoch}/{args.epochs}] cls loss: {cls_loss[0].avg:.3f}, {cls_loss[1].avg:.3f}, {cls_loss[2].avg:.3f}, {cls_loss[3].avg:.3f}, {cls_loss[4].avg:.3f}, {cls_loss[5].avg:.3f}, {cls_loss[6].avg:.3f}'

    #! m21-11-12, 注释
    # AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    #@ m21-11-12, 只记录七个分类器的准确率
    accs = Show_OnlyAccuracy(acc)
    writer.add_scalars('Accuracy/Test_Source', {'global': accs[0], 'left_eye':accs[1], 'right_eye':accs[2], 'nose':accs[3], 'left_mouse':accs[4], 'right_mouse':accs[5], 'global_local':accs[6]}, epoch)
    writer.add_scalars('Loss/Test_Source', {'global': cls_loss[0].avg, 'left_eye':cls_loss[1].avg, 'right_eye':cls_loss[2].avg, 'nose':cls_loss[3].avg, 'left_mouse':cls_loss[4].avg, 'right_mouse':cls_loss[5].avg, 'global_local':cls_loss[6].avg}, epoch)
    writer.add_scalars('Six_Probilities/Test_Source', {'situation_0': six_probilities[0].avg, 'situation_1':six_probilities[1].avg, 'situation_2':six_probilities[2].avg, 'situation_3':six_probilities[3].avg, 'situation_4':six_probilities[4].avg, 'situation_5':six_probilities[5].avg}, epoch)
    writer.add_scalars('Six_Accuracys/Test_Source', {'situation_0': six_accuracys[0].avg, 'situation_1':six_accuracys[1].avg, 'situation_2':six_accuracys[2].avg, 'situation_3':six_accuracys[3].avg, 'situation_4':six_accuracys[4].avg, 'situation_5':six_accuracys[5].avg}, epoch)

    LoggerInfo = '[Test (Source Domain) Epoch {0}]： Learning Rate {1} \n'\
    'Accuracy: global {Accuracys[0]:.4f}\t left_eye {Accuracys[1]:.4f}\t right_eye {Accuracys[2]:.4f}\t nose {Accuracys[3]:.4f}\t left_mouse {Accuracys[4]:.4f}\t right_mouse {Accuracys[5]:.4f}\t global_local {Accuracys[6]:.4f}\n'\
    'Cls_Loss: global {Loss[0].avg:.4f}\t left_eye {Loss[1].avg:.4f}\t right_eye {Loss[2].avg:.4f}\t nose {Loss[3].avg:.4f}\t left_mouse {Loss[4].avg:.4f}\t right_mouse {Loss[5].avg:.4f}\t global_local {Loss[6].avg:.4f}\n'\
    'Situ_Acc: Situation_0 {six_acc[0].avg:.4f}\t Situation_1 {six_acc[1].avg:.4f}\t Situation_2 {six_acc[2].avg:.4f}\t Situation_3 {six_acc[3].avg:.4f}\t Situation_4 {six_acc[4].avg:.4f}\t Situation_5 {six_acc[5].avg:.4f}\n'\
    'Situ_Pro: Situation_0 {six_prob[0].avg:.4f}\t Situation_1 {six_prob[1].avg:.4f}\t Situation_2 {six_prob[2].avg:.4f}\t Situation_3 {six_prob[3].avg:.4f}\t Situation_4 {six_prob[4].avg:.4f}\t Situation_5 {six_prob[5].avg:.4f}\n\n'\
    .format(epoch, args.lr, Accuracys=accs, Loss=cls_loss, six_acc=six_accuracys, six_prob=six_probilities)

    with open(args.OutputPath + "/test_result_source.log","a") as f:
        f.writelines(LoggerInfo)

    #@ Save Checkpoints
    classifier_name = {0:'global', 1:'left_eye', 2:'right_eye', 3:'nose', 4:'left_mouth', 5:'right_mouth', 6:'global_local'}
    best_classifier_id = accs.index(max(accs))
    best_classifier = classifier_name[best_classifier_id]
    best_acc = accs[best_classifier_id]
    if best_acc > Best_Acc and (confidence_target > 0.965 or epoch < 10):     # 根据Source Domain的效果判断是否存储
        Best_Acc = best_acc
        print("**************")
        print(f'[Save] Best Acc: {Best_Acc:.4f}, the classifier is {best_classifier}. Save the checkpoint! （Target Confidence is {confidence_target}）')
        print("**************")

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))
        else:
            torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))

    #@ ===========================================================================================
    
    # Test on Target Domain
    acc, prec, recall = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    cls_loss, loss, afn_loss, data_time, batch_time = [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities, six_accuracys = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]

    end = time.time()
    test_target_bar = tqdm(test_target_dataloader)
    for step, (_, input, landmark, label) in enumerate(test_target_bar):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        data_time.update(time.time() - end)
        
        # Forward Propagation
        with torch.no_grad():
            end = time.time()
            #! m21-11-12，注释
            # feature, output, loc_output = model(input, landmark)
            #@ m21-11-11, 新的model调用
            features, preds = model(input, landmark)
            batch_time.update(time.time()-end)
        
        # Compute Loss
        #! m21-11-11, 注释
        '''
            global_cls_loss_ = nn.CrossEntropyLoss()(output, label)
            local_cls_loss_ = nn.CrossEntropyLoss()(loc_output, label) if args.useLocalFeature else 0
            afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN == 'HAFN' else SAFN(feature, args.weight_L2norm, args.deltaRadius)) if args.useAFN else 0
            loss_ = global_cls_loss_ + local_cls_loss_ + (afn_loss_ if args.useAFN else 0)
        '''

        #@ m21-11-12, 新的计算loss的方式
        loss_ = 0
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i], label)
            cls_loss[i].update(tmp.cpu().data.item(), input.size(0))
            loss_ += tmp

        # Compute accuracy, precision and recall
        for classifier_id in range(7): # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id], label, acc[classifier_id], prec[classifier_id], recall[classifier_id])

        Count_Probility_Accuracy(six_probilities, six_accuracys, preds, label)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        end = time.time()
        test_target_bar.desc = f'[Test (Target Domain) Epoch {epoch}/{args.epochs}] cls loss: {cls_loss[0].avg:.3f}, {cls_loss[1].avg:.3f}, {cls_loss[2].avg:.3f}, {cls_loss[3].avg:.3f}, {cls_loss[4].avg:.3f}, {cls_loss[5].avg:.3f}, {cls_loss[6].avg:.3f}'

    #! m21-11-12, 注释
    # AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    #@ m21-11-12, 只记录七个分类器的准确率
    accs = Show_OnlyAccuracy(acc)
    writer.add_scalars('Accuracy/Test_Target', {'global': accs[0], 'left_eye':accs[1], 'right_eye':accs[2], 'nose':accs[3], 'left_mouse':accs[4], 'right_mouse':accs[5], 'global_local':accs[6]}, epoch)
    writer.add_scalars('Loss/Test_Target', {'global': cls_loss[0].avg, 'left_eye':cls_loss[1].avg, 'right_eye':cls_loss[2].avg, 'nose':cls_loss[3].avg, 'left_mouse':cls_loss[4].avg, 'right_mouse':cls_loss[5].avg, 'global_local':cls_loss[6].avg}, epoch)
    writer.add_scalars('Six_Probilities/Test_Target', {'situation_0': six_probilities[0].avg, 'situation_1':six_probilities[1].avg, 'situation_2':six_probilities[2].avg, 'situation_3':six_probilities[3].avg, 'situation_4':six_probilities[4].avg, 'situation_5':six_probilities[5].avg}, epoch)
    writer.add_scalars('Six_Accuracys/Test_Target', {'situation_0': six_accuracys[0].avg, 'situation_1':six_accuracys[1].avg, 'situation_2':six_accuracys[2].avg, 'situation_3':six_accuracys[3].avg, 'situation_4':six_accuracys[4].avg, 'situation_5':six_accuracys[5].avg}, epoch)

    LoggerInfo = '[Test (Target Domain) Epoch {0}]： Learning Rate {1} \n'\
    'Accuracy: global {Accuracys[0]:.4f}\t left_eye {Accuracys[1]:.4f}\t right_eye {Accuracys[2]:.4f}\t nose {Accuracys[3]:.4f}\t left_mouse {Accuracys[4]:.4f}\t right_mouse {Accuracys[5]:.4f}\t global_local {Accuracys[6]:.4f}\n'\
    'Cls_Loss: global {Loss[0].avg:.4f}\t left_eye {Loss[1].avg:.4f}\t right_eye {Loss[2].avg:.4f}\t nose {Loss[3].avg:.4f}\t left_mouse {Loss[4].avg:.4f}\t right_mouse {Loss[5].avg:.4f}\t global_local {Loss[6].avg:.4f}\n'\
    'Situ_Acc: Situation_0 {six_acc[0].avg:.4f}\t Situation_1 {six_acc[1].avg:.4f}\t Situation_2 {six_acc[2].avg:.4f}\t Situation_3 {six_acc[3].avg:.4f}\t Situation_4 {six_acc[4].avg:.4f}\t Situation_5 {six_acc[5].avg:.4f}\n'\
    'Situ_Pro: Situation_0 {six_prob[0].avg:.4f}\t Situation_1 {six_prob[1].avg:.4f}\t Situation_2 {six_prob[2].avg:.4f}\t Situation_3 {six_prob[3].avg:.4f}\t Situation_4 {six_prob[4].avg:.4f}\t Situation_5 {six_prob[5].avg:.4f}\n\n'\
    .format(epoch, args.lr, Accuracys=accs, Loss=cls_loss, six_acc=six_accuracys, six_prob=six_probilities)

    with open(args.OutputPath + "/test_result_target.log","a") as f:
        f.writelines(LoggerInfo)

    return Best_Acc

def main():
    """Main."""
 
    # Parse Argument
    args = construct_args()         # 构造参数
    torch.manual_seed(args.seed)    # 人工种子
    folder = str(int(time.time()))
    print(f"Timestamp is {folder}")
    args.OutputPath = os.path.join(args.OutputPath, folder)
    makeFolder(args)

    # Experiment Information
    print('Log Name: %s' % args.Log_Name)
    print('Output Path: %s' % args.OutputPath)
    print('Backbone: %s' % args.Backbone)
    print('Resume Model: %s' % args.Resume_Model)
    print('CUDA_VISIBLE_DEVICES: %s' % args.GPU_ID)

    print('================================================')

    print('Use {} * {} Image'.format(args.faceScale, args.faceScale))
    print('SourceDataset: %s' % args.sourceDataset)
    print('TargetDataset: %s' % args.targetDataset)
    print('Train Batch Size: %d' % args.train_batch_size)
    print('Test Batch Size: %d' % args.test_batch_size)

    print('================================================')

    if args.showFeature:
        print('Show Visualiza Result of Feature.')

    if args.isTest:# 只是测试一下模型的性能
        print('Test Model.')
    else: # 正常的训练，打印训练参数
        print('Train Epoch: %d' % args.epochs)
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Weight Decay: %f' % args.weight_decay)

        if args.useAFN: # AFN的方法
            print('Use AFN Loss: %s' % args.methodOfAFN)
            if args.methodOfAFN == 'HAFN':    # hard afn
                print('Radius of HAFN Loss: %f' % args.radius)
            else:                           # soft afn
                print('Delta Radius of SAFN Loss: %f' % args.deltaRadius)
            print('Weight L2 nrom of AFN Loss: %f' % args.weight_L2norm)

    print('================================================')

    print('Number of classes : %d' % args.class_num) # 表情的类别数
    if not args.useLocalFeature:
        print('Only use global feature.') # 只使用全局特征
    else:
        print('Use global feature and local feature.')

        if args.useIntraGCN:
            print('Use Intra GCN.') # 是否使用域内GCN进行传播
        if args.useInterGCN:
            print('Use Inter GCN.') # 是否使用域间GCN进行传播

        if args.useRandomMatrix and args.useAllOneMatrix:
            print('Wrong : Use RandomMatrix and AllOneMatrix both!')
            return None
        elif args.useRandomMatrix:
            print('Use Random Matrix in GCN.')
        elif args.useAllOneMatrix:
            print('Use All One Matrix in GCN.')

        if args.useCov and args.useCluster: # 使用协方差矩阵进行初始化or采用k-means算法进行初始化
            print('Wrong : Use Cov and Cluster both!')
            return None
        else:
            if args.useCov:
                print('Use Mean and Cov.') #todo: mean是指什么？
            else:
                print('Use Mean.') if not args.useCluster else print('Use Mean in Cluster.')

    print('================================================')

    print('================================================')
    # Bulid Model
    print('Building Model...')
    model = BulidModel(args)

    with open(args.OutputPath + "/first.log", "a") as f:
        num = 0
        for k, v in model.named_parameters():
            num += 1
            f.writelines(str(num) + "、" + str(k) + "\n" + str(v) + "\n\n")
    print('Done!')
    print('================================================')


    # Bulid Dataloder
    print("Building Train and Test Dataloader...")
    train_source_dataloader, _ = BulidDataloader(args, flag1='train', flag2='source') # 构建了source的源域数据集生成器。调用他之后能够返回batch size数量的（裁剪后的人脸图片，人脸的五个关键点，表情标签）
    train_target_dataloader, init_train_dataset_data = BulidDataloader(args, flag1='train', flag2='target') # 构建了训练的target目标域的数据集生成器，调用他之后能够返回batch size数量的（裁剪后的人脸图片，人脸的五个关键点，表情标签）
    test_source_dataloader, _ = BulidDataloader(args, flag1='test', flag2='source')   # test跟train的数据集生成器只有图像的预处理那里会有点不同
    test_target_dataloader, _ = BulidDataloader(args, flag1='test', flag2='target')

    print('Done!')

    #  Set Optimizer
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(args, model) # 待优化的参数
    optimizer = Set_Optimizer(args, param_optim, args.lr, args.weight_decay, args.momentum)
    print('Done!')

    print('================================================')

    # Save Best Checkpoint
    Best_Acc = 0
    confidence = 0

    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter(os.path.join(args.OutputPath, 'visual_board'))


    for epoch in range(1, args.epochs + 1):
        if args.showFeature and epoch % 5 == 1:
            print(f"=================\ndraw the tSNE graph...")
            Visualization(args.OutputPath + '/result_pics/train/source/{}_Source.jpg'.format(epoch), model, dataloader=train_source_dataloader, useClassify=True, domain='Source')
            Visualization(args.OutputPath + '/result_pics/train/target/{}_Target.jpg'.format(epoch), model, train_target_dataloader, useClassify=True, domain='Target')

            VisualizationForTwoDomain(args.OutputPath + '/result_pics/train_tow_domain/{}_train'.format(epoch), model, train_source_dataloader, train_target_dataloader, useClassify=True, showClusterCenter=False)
            VisualizationForTwoDomain(args.OutputPath + '/result_pics/test_tow_domain/{}_test'.format(epoch), model, test_source_dataloader, test_target_dataloader, useClassify=True, showClusterCenter=False)
            print(f"finish drawing!\n=================")

        if not args.isTest:
            if args.useCluster and epoch % 5 == 1:
                print(f"=================\nupdate the running_mean...")
                Initialize_Mean_Cluster(args, model, True, train_source_dataloader, train_target_dataloader)
                torch.cuda.empty_cache()
                print(f"finish the updating!\n=================")
            Train(args, model, train_source_dataloader, optimizer, epoch, writer)
            
            if epoch >= 10:
                _, confidence, probility = BuildLabeledDataloader(args, train_target_dataloader, init_train_dataset_data, model)
        Best_Acc = Test(args, model, test_source_dataloader, test_target_dataloader, Best_Acc, epoch, confidence, writer)

        torch.cuda.empty_cache()

    writer.close()

if __name__ == '__main__':
    main()
