import os
import sys
import time
import tqdm
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Loss import Entropy, DANN, CDAN, HAFN, SAFN
from Utils import *

def construct_args():
    parser = argparse.ArgumentParser(description='Domain adaptation for Expression Classification')

    parser.add_argument('--Log_Name', type=str, help='Log Name')
    parser.add_argument('--OutputPath', type=str, help='Output Path')
    parser.add_argument('--Backbone', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
    parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='None')
    parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

    parser.add_argument('--useDAN', type=str2bool, default=False, help='whether to use DAN Loss')
    parser.add_argument('--methodOfDAN', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN'])

    parser.add_argument('--useAFN', type=str2bool, default=False, help='whether to use AFN Loss')
    parser.add_argument('--methodOfAFN', type=str, default='SAFN', choices=['HAFN', 'SAFN'])
    parser.add_argument('--radius', type=float, default=25.0, help='radius of HAFN (default: 25.0)')
    parser.add_argument('--deltaRadius', type=float, default=1.0, help='radius of SAFN (default: 1.0)')
    parser.add_argument('--weight_L2norm', type=float, default=0.05, help='weight L2 norm of AFN (default: 0.05)')

    parser.add_argument('--faceScale', type=int, default=112, help='Scale of face (default: 112)')
    parser.add_argument('--sourceDataset', type=str, default='RAF', choices=['RAF', 'AFED', 'MMI','FER2013'])
    parser.add_argument('--targetDataset', type=str, default='CK+',
                        choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED'])
    parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=64, help='input batch size for testing (default: 64)')
    parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_ad', type=float, default=0.01)

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay (default: 0.0005)')

    parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')
    parser.add_argument('--showFeature', type=str2bool, default=False, help='whether to show feature')

    parser.add_argument('--useIntraGCN', type=str2bool, default=False, help='whether to use Intra-GCN')
    parser.add_argument('--useInterGCN', type=str2bool, default=False, help='whether to use Inter-GCN')
    parser.add_argument('--useLocalFeature', type=str2bool, default=False, help='whether to use Local Feature')

    parser.add_argument('--useRandomMatrix', type=str2bool, default=False, help='whether to use Random Matrix')
    parser.add_argument('--useAllOneMatrix', type=str2bool, default=False, help='whether to use All One Matrix')

    parser.add_argument('--useCov', type=str2bool, default=False, help='whether to use Cov')
    parser.add_argument('--useCluster', type=str2bool, default=False, help='whether to use Cluster')

    parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num_divided', type=int, default=10, help='the number of blocks [0, loge(7)] to be divided')
    parser.add_argument('--randomLayer', type=str2bool, default=False, help='whether to use random')
    
    parser.add_argument('--target_loss_ratio', type=int, default=2, help='the ratio of seven classifier using on target label on the base of classifier_loss_ratio')

    args = parser.parse_args()   # 构造参数
    return args


def Train(args, model, ad_nets, random_layers, train_source_dataloader, train_target_dataloader, labeled_train_target_dataloader, optimizer, optimizer_ad,
          epoch, writer):
    """Train."""
    model.train()
    # torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    cls_loss, dan_loss, loss, data_time, batch_time = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter()
    data_time, batch_time = AverageMeter(), AverageMeter()
    num_ADNets = [0 for i in range(7)]
    six_probilities_source, six_accuracys_source = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]
    six_probilities_target, six_accuracys_target = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]

    #@ 特别记录一下train中的target
    acc_target, prec_target, recall_target = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]

    #@ 在七个分类器的预测值都相同的情况下再增加多一个entropy的condition（这些统计参数只用于target domain）
    # delta = 1.9459 / args.num_divided / args.num_divided
    delta = 0.19459 / args.num_divided
    entropy_thresholds = np.arange(delta, 0.19459 + delta, delta)
    probilities_entropy, accuracys_entropy = [AverageMeter() for i in range(args.num_divided)], [AverageMeter() for i in range(args.num_divided)]
    
    #@ 统计一下在不同domain上训练的loss
    source_cls_loss, target_cls_loss = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]

    if args.useDAN:
        num_ADNet = 0
        [ad_nets[i].train() for i in range(len(ad_nets))]

    # Decay Learn Rate per Epoch
    if epoch <= 30:
        args.lr, args.lr_ad = 1e-5, 0.0001
    elif epoch <= 50:
        args.lr, args.lr_ad = 5e-6, 0.0001
    else:
        args.lr, args.lr_ad = 1e-6, 0.00001

    # if epoch <= 10:
    #     args.lr, args.lr_ad = 1e-5, 0.0001
    # elif epoch <= 40:
    #     args.lr, args.lr_ad = 1e-6, 0.0001
    # else:
    #     args.lr, args.lr_ad = 1e-7, 0.00001

    optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr)
    if args.useDAN:
        optimizer_ad, lr_ad = lr_scheduler_withoutDecay(optimizer_ad, lr=args.lr_ad)

    # Get Source/Target Dataloader iterator
    iter_source_dataloader = iter(train_source_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)
    if labeled_train_target_dataloader != None:
        iter_labeled_target_dataloader = iter(labeled_train_target_dataloader)

    # len(data_loader) = math.ceil(len(data_loader.dataset)/batch_size)
    num_iter = len(train_source_dataloader) if (len(train_source_dataloader) > len(train_target_dataloader)) else len(train_target_dataloader)

    end = time.time()
    train_bar = tqdm(range(num_iter))
    for step, batch_index in enumerate(train_bar):
        try:
            _, data_source, landmark_source, label_source = iter_source_dataloader.next()
        except:
            iter_source_dataloader = iter(train_source_dataloader)
            _, data_source, landmark_source, label_source = iter_source_dataloader.next()
        try:
            _, data_target, landmark_target, label_target = iter_target_dataloader.next()
        except:
            iter_target_dataloader = iter(train_target_dataloader)
            _, data_target, landmark_target, label_target = iter_target_dataloader.next()
        
        
        data_time.update(time.time() - end)

        data_source, landmark_source, label_source = data_source.cuda(), landmark_source.cuda(), label_source.cuda()
        data_target, landmark_target, label_target = data_target.cuda(), landmark_target.cuda(), label_target.cuda()

        # Forward Propagation
        end = time.time()
        #! m21-11-13, 注释
        # feature, output, loc_output = model(torch.cat((data_source, data_target), 0), torch.cat((landmark_source, landmark_target), 0))
        features, preds = model(torch.cat((data_source, data_target), 0), torch.cat((landmark_source, landmark_target), 0))
        batch_time.update(time.time() - end)

        # Compute Loss
        '''
            global_cls_loss_ = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source) # 这里计算交叉熵的时候都是只用了Source Domain的部分和Label
            local_cls_loss_ = nn.CrossEntropyLoss()(loc_output.narrow(0, 0, data_source.size(0)), label_source) if args.useLocalFeature else 0
            afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN == 'HAFN' else SAFN(feature, args.weight_L2norm, args.deltaRadius)) if args.useAFN else 0
        '''

        #@ m21-11-13, Compute Classifier Loss(Source Domain)
        classifiers_loss_ratio = [7, 1, 1, 1, 1, 1, 7]
        loss_ = 0
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i].narrow(0, 0, data_source.size(0)), label_source)
            cls_loss[i].update(tmp.cpu().data.item(), data_source.size(0))
            source_cls_loss[i].update(tmp, data_source.size(0))
            loss_ += classifiers_loss_ratio[i] * tmp

        
        #@ using the fake label of target domain to train the classifier
        if labeled_train_target_dataloader != None:
            try:
                _, data_labeled_target, landmark_labeled_target, label_labeled_target = iter_labeled_target_dataloader.next()
            except:
                iter_labeled_target_dataloader = iter(labeled_train_target_dataloader)
                _, data_labeled_target, landmark_labeled_target, label_labeled_target = iter_labeled_target_dataloader.next()
            
            data_labeled_target, landmark_labeled_target, label_labeled_target = data_labeled_target.cuda(), landmark_labeled_target.cuda(), label_labeled_target.cuda()
            features_faked, preds_faked = model(data_labeled_target, landmark_labeled_target)
            criteria = nn.CrossEntropyLoss()
            ratio = len(train_source_dataloader) / len(labeled_train_target_dataloader)
            for i in range(7):
                tmp = criteria(preds_faked[i], label_labeled_target)
                cls_loss[i].update(tmp.cpu().data.item(), data_labeled_target.size(0))
                target_cls_loss[i].update(tmp, data_labeled_target.size(0))
                # loss_ += args.target_loss_ratio * classifiers_loss_ratio[i] * tmp
                loss_ += tmp

        #@ m21-11-13, Compute the DAN Loss, 七个
        dan_loss_ = 0
        dan_idx = [0, 1, 1, 1, 1, 1, 2]
        softmax = nn.Softmax(dim=1)
        if args.useDAN: # 这里在利用对抗学习的时候，是没有用到标签的，无监督学习
            for classifier_id in range(7):
                tmp = 0
                softmax_output = softmax(preds[classifier_id])
                if args.methodOfDAN == 'CDAN-E':
                    entropy = Entropy(softmax_output)
                    tmp = CDAN([features[classifier_id], softmax_output], ad_nets[dan_idx[classifier_id]], entropy, calc_coeff(num_iter * (epoch - 1) + batch_index), random_layers[dan_idx[classifier_id]])
                    dan_loss_ += tmp
                    dan_loss[classifier_id].update(tmp, features[classifier_id].size(0))
                elif args.methodOfDAN == 'CDAN':
                    dan_loss_ = CDAN([feature, softmax_output], ad_net, None, None, random_layer)
                elif args.methodOfDAN == 'DANN':
                    dan_loss_ = DANN(feature, ad_net)
        else:
            dan_loss_ = 0

        if args.useAFN:
            loss_ += afn_loss_

        if args.useDAN:
            loss_ += dan_loss_

        # Log Adversarial Network Accuracy
        for classifier_id in range(7):
            if args.useDAN:
                if args.methodOfDAN == 'CDAN' or args.methodOfDAN == 'CDAN-E':
                    softmax_output = nn.Softmax(dim=1)(preds[classifier_id])
                    if args.randomLayer:
                        random_out = random_layers[dan_idx[classifier_id]].forward([features[classifier_id], softmax_output])
                        adnet_output = ad_nets[dan_idx[classifier_id]](random_out.view(-1, random_out.size(1)))
                    else:
                        op_out = torch.bmm(softmax_output.unsqueeze(2), features[classifier_id].unsqueeze(1)) # softmax_output's shape is (batchSize, 7, 1) feature's shape is (batchSize, 1, 384)
                        adnet_output = ad_nets[dan_idx[classifier_id]](op_out.view(-1, softmax_output.size(1) * features[classifier_id].size(1)))
                elif args.methodOfDAN == 'DANN':
                    adnet_output = ad_net(feature)

                adnet_output = adnet_output.cpu().data.numpy()
                adnet_output[adnet_output > 0.5] = 1
                adnet_output[adnet_output <= 0.5] = 0
                num_ADNets[classifier_id] += np.sum(adnet_output[:args.train_batch_size]) + (args.train_batch_size - np.sum(adnet_output[args.train_batch_size:]))

        # Back Propagation
        optimizer.zero_grad()
        if args.useDAN:
            optimizer_ad.zero_grad()

        # with torch.autograd.detect_anomaly():
        loss_.backward()

        optimizer.step()

        if args.useDAN:
            optimizer_ad.step()

        # Compute accuracy, precision and recall
        for classifier_id in range(7): # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id].narrow(0, 0, data_source.size(0)), label_source, acc[classifier_id], prec[classifier_id], recall[classifier_id])

        # Compute accuracy, precision and recall
        for classifier_id in range(7): # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id].narrow(0, data_target.size(0), data_target.size(0)), label_target, acc_target[classifier_id], prec_target[classifier_id], recall_target[classifier_id])

        Count_Probility_Accuracy(six_probilities_source, six_accuracys_source, [pred.narrow(0, 0, data_source.size(0)) for pred in preds], label_source)
        Count_Probility_Accuracy(six_probilities_target, six_accuracys_target, [pred.narrow(0, data_target.size(0), data_target.size(0)) for pred in preds], label_target)

        #@ 这里只记录一下target的情况就可以了
        Count_Probility_Accuracy_Entropy(entropy_thresholds, probilities_entropy, accuracys_entropy, [pred.narrow(0, data_target.size(0), data_target.size(0)) for pred in preds], label_target)

        # Log loss
        loss.update(float(loss_.cpu().data.item()), data_source.size(0))
        end = time.time()
        train_bar.desc = f'[Train Epoch {epoch}/{args.epochs}] dan_loss: {dan_loss[0].avg:.3f}, {dan_loss[1].avg:.3f}, {dan_loss[2].avg:.3f}, {dan_loss[3].avg:.3f}, {dan_loss[4].avg:.3f}, {dan_loss[5].avg:.3f}, {dan_loss[6].avg:.3f}'

    #! m21-11-12, 注释
    # AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    #@ m21-11-12, 只记录七个分类器的准确率
    accs = Show_OnlyAccuracy(acc)
    
    accs_target = Show_OnlyAccuracy(acc_target)
    
    writer.add_scalars('Train/CLS_Acc_Source', {'global': accs[0], 'left_eye':accs[1], 'right_eye':accs[2], 'nose':accs[3], 'left_mouse':accs[4], 'right_mouse':accs[5], 'global_local':accs[6]}, epoch)
    writer.add_scalars('Train/CLS_Loss_Source', {'global': cls_loss[0].avg, 'left_eye':cls_loss[1].avg, 'right_eye':cls_loss[2].avg, 'nose':cls_loss[3].avg, 'left_mouse':cls_loss[4].avg, 'right_mouse':cls_loss[5].avg, 'global_local':cls_loss[6].avg}, epoch)
    # writer.add_scalars('Train/Six_Probilities_Source', {'Situation_0': six_probilities_source[0].avg, 'Situation_1':six_probilities_source[1].avg, 'Situation_2':six_probilities_source[2].avg, 'Situation_3':six_probilities_source[3].avg, 'Situation_4':six_probilities_source[4].avg, 'Situation_5':six_probilities_source[5].avg}, epoch)
    # writer.add_scalars('Train/Six_Accuracys_Source', {'Situation_0': six_accuracys_source[0].avg, 'Situation_1':six_accuracys_source[1].avg, 'Situation_2':six_accuracys_source[2].avg, 'Situation_3':six_accuracys_source[3].avg, 'Situation_4':six_accuracys_source[4].avg, 'Situation_5':six_accuracys_source[5].avg}, epoch)
    # writer.add_scalars('Train/Six_Probilities_Target', {'Situation_0': six_probilities_target[0].avg, 'Situation_1':six_probilities_target[1].avg, 'Situation_2':six_probilities_target[2].avg, 'Situation_3':six_probilities_target[3].avg, 'Situation_4':six_probilities_target[4].avg, 'Situation_5':six_probilities_target[5].avg}, epoch)
    # writer.add_scalars('Train/Six_Accuracys_Target', {'Situation_0': six_accuracys_target[0].avg, 'Situation_1':six_accuracys_target[1].avg, 'Situation_2':six_accuracys_target[2].avg, 'Situation_3':six_accuracys_target[3].avg, 'Situation_4':six_accuracys_target[4].avg, 'Situation_5':six_accuracys_target[5].avg}, epoch)
    
    writer.add_scalars('Train/Merely_Source_CLS_Loss', {'global': source_cls_loss[0].avg, 'left_eye':source_cls_loss[1].avg, 'right_eye':source_cls_loss[2].avg, 'nose':source_cls_loss[3].avg, 'left_mouse':source_cls_loss[4].avg, 'right_mouse':source_cls_loss[5].avg, 'global_local':source_cls_loss[6].avg}, epoch)
    writer.add_scalars('Train/Merely_Target_CLS_Loss', {'global': target_cls_loss[0].avg, 'left_eye':target_cls_loss[1].avg, 'right_eye':target_cls_loss[2].avg, 'nose':target_cls_loss[3].avg, 'left_mouse':target_cls_loss[4].avg, 'right_mouse':target_cls_loss[5].avg, 'global_local':target_cls_loss[6].avg}, epoch)
    
    writer.add_scalars('Train/CLS_Acc_Target', {'global': accs_target[0], 'left_eye':accs_target[1], 'right_eye':accs_target[2], 'nose':accs_target[3], 'left_mouse':accs_target[4], 'right_mouse':accs_target[5], 'global_local':accs_target[6]}, epoch)
    acc_dic, pro_dic = {}, {}
    for i in range(args.num_divided):
        acc_dic.update({'entropy_'+str(entropy_thresholds[i]): accuracys_entropy[i].avg})
        pro_dic.update({'entropy_'+str(entropy_thresholds[i]): probilities_entropy[i].avg})
    writer.add_scalars('Train/Accuracys_Entropy', acc_dic, epoch)
    writer.add_scalars('Train/Probility_Entropy', pro_dic, epoch)
    if args.useDAN:
        dan_accs = []
        for classifier_id in range(7):
            dan_accs.append(num_ADNets[classifier_id] / (2.0 * args.train_batch_size * num_iter))
        writer.add_scalars('Train/DAN_Acc', {'global':dan_accs[0], 'left_eye':dan_accs[1], 'right_eye':dan_accs[2], 'nose':dan_accs[3], 'left_mouse':dan_accs[4], 'right_mouse':dan_accs[5], 'global_local':dan_accs[6]}, epoch)
        writer.add_scalars('Train/DAN_Loss', {'global':dan_loss[0].avg, 'left_eye':dan_loss[1].avg, 'right_eye':dan_loss[2].avg, 'nose':dan_loss[3].avg, 'left_mouse':dan_loss[4].avg, 'right_mouse':dan_loss[5].avg, 'global_local':dan_loss[6].avg}, epoch)

    LoggerInfo = '[Train Epoch {0}]： Learning Rate {1}  DAN Learning Rate {2}\n'\
    'CLS_Acc:  global {cls_accs[0]:.4f}\t left_eye {cls_accs[1]:.4f}\t right_eye {cls_accs[2]:.4f}\t nose {cls_accs[3]:.4f}\t left_mouse {cls_accs[4]:.4f}\t right_mouse {cls_accs[5]:.4f}\t global_local {cls_accs[6]:.4f}\n'\
    'DAN_ACC:  global {dan_accs[0]:.4f}\t left_eye {dan_accs[1]:.4f}\t right_eye {dan_accs[2]:.4f}\t nose {dan_accs[3]:.4f}\t left_mouse {dan_accs[4]:.4f}\t right_mouse {dan_accs[5]:.4f}\t global_local {dan_accs[6]:.4f}\n'\
    'SUM_Loss: global {cls_loss[0].avg:.4f}\t left_eye {cls_loss[1].avg:.4f}\t right_eye {cls_loss[2].avg:.4f}\t nose {cls_loss[3].avg:.4f}\t left_mouse {cls_loss[4].avg:.4f}\t right_mouse {cls_loss[5].avg:.4f}\t global_local {cls_loss[6].avg:.4f}\n'\
    'DAN_Loss: global {dan_loss[0].avg:.4f}\t left_eye {dan_loss[1].avg:.4f}\t right_eye {dan_loss[2].avg:.4f}\t nose {dan_loss[3].avg:.4f}\t left_mouse {dan_loss[4].avg:.4f}\t right_mouse {dan_loss[5].avg:.4f}\t global_local {dan_loss[6].avg:.4f}\n'\
    'Situ_Acc_Source: Situation_0 {six_acc_source[0].avg:.4f}\t Situation_1 {six_acc_source[1].avg:.4f}\t Situation_2 {six_acc_source[2].avg:.4f}\t Situation_3 {six_acc_source[3].avg:.4f}\t Situation_4 {six_acc_source[4].avg:.4f}\t Situation_5 {six_acc_source[5].avg:.4f}\n'\
    'Situ_Pro_Source: Situation_0 {six_prob_source[0].avg:.4f}\t Situation_1 {six_prob_source[1].avg:.4f}\t Situation_2 {six_prob_source[2].avg:.4f}\t Situation_3 {six_prob_source[3].avg:.4f}\t Situation_4 {six_prob_source[4].avg:.4f}\t Situation_5 {six_prob_source[5].avg:.4f}\n'\
    'Situ_Acc_Target: Situation_0 {six_acc_target[0].avg:.4f}\t Situation_1 {six_acc_target[1].avg:.4f}\t Situation_2 {six_acc_target[2].avg:.4f}\t Situation_3 {six_acc_target[3].avg:.4f}\t Situation_4 {six_acc_target[4].avg:.4f}\t Situation_5 {six_acc_target[5].avg:.4f}\n'\
    'Situ_Pro_Target: Situation_0 {six_prob_target[0].avg:.4f}\t Situation_1 {six_prob_target[1].avg:.4f}\t Situation_2 {six_prob_target[2].avg:.4f}\t Situation_3 {six_prob_target[3].avg:.4f}\t Situation_4 {six_prob_target[4].avg:.4f}\t Situation_5 {six_prob_target[5].avg:.4f}\n'\
    'Source_CLS_Loss: global {source_cls_loss[0].avg:.4f}\t left_eye {source_cls_loss[1].avg:.4f}\t right_eye {source_cls_loss[2].avg:.4f}\t nose {source_cls_loss[3].avg:.4f}\t left_mouse {source_cls_loss[4].avg:.4f}\t right_mouse {source_cls_loss[5].avg:.4f}\t global_local {source_cls_loss[6].avg:.4f}\n'\
    'Target_CLS_Loss: global {target_cls_loss[0].avg:.4f}\t left_eye {target_cls_loss[1].avg:.4f}\t right_eye {target_cls_loss[2].avg:.4f}\t nose {target_cls_loss[3].avg:.4f}\t left_mouse {target_cls_loss[4].avg:.4f}\t right_mouse {target_cls_loss[5].avg:.4f}\t global_local {target_cls_loss[6].avg:.4f}\n\n'\
    .format(epoch, args.lr, args.lr_ad, cls_accs=accs, dan_accs=dan_accs, cls_loss=cls_loss, dan_loss=dan_loss, six_acc_source=six_accuracys_source, six_prob_source=six_probilities_source, six_acc_target=six_accuracys_target, six_prob_target=six_probilities_target, source_cls_loss=source_cls_loss, target_cls_loss=target_cls_loss)

    with open(args.OutputPath + "/train_result_transfer.log", "a") as f:
        f.writelines(LoggerInfo)


def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Acc, epoch, writer):
    """Test."""

    model.eval()
    # torch.autograd.set_detect_anomaly(True)

    #! m21-11-13, 注释
    # iter_source_dataloader = iter(test_source_dataloader)
    # iter_target_dataloader = iter(test_target_dataloader)

    # Test on Source Domain
    acc, prec, recall = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    cls_loss, loss, afn_loss, data_time, batch_time = [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities, six_accuracys = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]
    acc_glg, prec_glg, recall_glg = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)] # sum prediction from global and global with local

    end = time.time()
    test_source_bar = tqdm(test_source_dataloader)
    for batch_index, (_, input, landmark, target) in enumerate(test_source_bar):
        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()
        data_time.update(time.time() - end)

        # Forward Propagation
        with torch.no_grad():
            end = time.time()
            #! m21-11-12，注释
            # feature, output, loc_output = model(input, landmark)
            #@ m21-11-13, 新的model调用
            features, preds = model(input, landmark)
            batch_time.update(time.time() - end)

        #@ m21-11-13, 新的计算loss的方式
        loss_ = 0
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i], target)
            cls_loss[i].update(tmp.cpu().data.item(), input.size(0))
            loss_ += tmp
        
        # Compute accuracy, precision and recall
        for classifier_id in range(7): # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id], target, acc[classifier_id], prec[classifier_id], recall[classifier_id])
        Compute_Accuracy(args, preds[0] + preds[6], target, acc_glg, prec_glg, recall_glg)
        
        # Count the six statisc probility and accuracy
        Count_Probility_Accuracy(six_probilities, six_accuracys, preds, target)

        # Log loss
        loss.update(float(loss_.cpu().data.numpy()))
        end = time.time()
        test_source_bar.desc = f'[Test (Source Domain) Epoch {epoch}/{args.epochs}] cls loss: {cls_loss[0].avg:.3f}, {cls_loss[1].avg:.3f}, {cls_loss[2].avg:.3f}, {cls_loss[3].avg:.3f}, {cls_loss[4].avg:.3f}, {cls_loss[5].avg:.3f}, {cls_loss[6].avg:.3f}'
    
    #! m21-11-12, 注释
    # AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    #@ m21-11-12, 只记录七个分类器的准确率
    accs = Show_OnlyAccuracy(acc)

    #@ m21-11-30, 计算glg的acc
    acc_glg_value = 0
    for class_id in range(7):
        acc_glg_value += acc_glg[class_id].avg
    acc_glg_value /= 7
        
    writer.add_scalars('Test/CLS_Acc_Source', {'global': accs[0], 'left_eye':accs[1], 'right_eye':accs[2], 'nose':accs[3], 'left_mouse':accs[4], 'right_mouse':accs[5], 'global_local':accs[6], 'global_local_global':acc_glg_value}, epoch)
    writer.add_scalars('Test/CLS_Loss_Source', {'global': cls_loss[0].avg, 'left_eye':cls_loss[1].avg, 'right_eye':cls_loss[2].avg, 'nose':cls_loss[3].avg, 'left_mouse':cls_loss[4].avg, 'right_mouse':cls_loss[5].avg, 'global_local':cls_loss[6].avg}, epoch)
    writer.add_scalars('Test/Six_Probilities_Source', {'situation_0': six_probilities[0].avg, 'situation_1':six_probilities[1].avg, 'situation_2':six_probilities[2].avg, 'situation_3':six_probilities[3].avg, 'situation_4':six_probilities[4].avg, 'situation_5':six_probilities[5].avg}, epoch)
    writer.add_scalars('Test/Six_Accuracys_Source', {'situation_0': six_accuracys[0].avg, 'situation_1':six_accuracys[1].avg, 'situation_2':six_accuracys[2].avg, 'situation_3':six_accuracys[3].avg, 'situation_4':six_accuracys[4].avg, 'situation_5':six_accuracys[5].avg}, epoch)

    #todo: 更改这里的loggerInfo
    LoggerInfo = '[Test (Source Domain) Epoch {0}]： Learning Rate {1} \n'\
    'Accuracy: global {Accuracys[0]:.4f}\t left_eye {Accuracys[1]:.4f}\t right_eye {Accuracys[2]:.4f}\t nose {Accuracys[3]:.4f}\t left_mouse {Accuracys[4]:.4f}\t right_mouse {Accuracys[5]:.4f}\t global_local {Accuracys[6]:.4f} global_local_global {acc_glg_value:.4f}\n'\
    'Cls_Loss: global {Loss[0].avg:.4f}\t left_eye {Loss[1].avg:.4f}\t right_eye {Loss[2].avg:.4f}\t nose {Loss[3].avg:.4f}\t left_mouse {Loss[4].avg:.4f}\t right_mouse {Loss[5].avg:.4f}\t global_local {Loss[6].avg:.4f}\n'\
    'Situ_Acc: Situation_0 {six_acc[0].avg:.4f}\t Situation_1 {six_acc[1].avg:.4f}\t Situation_2 {six_acc[2].avg:.4f}\t Situation_3 {six_acc[3].avg:.4f}\t Situation_4 {six_acc[4].avg:.4f}\t Situation_5 {six_acc[5].avg:.4f}\n'\
    'Situ_Pro: Situation_0 {six_prob[0].avg:.4f}\t Situation_1 {six_prob[1].avg:.4f}\t Situation_2 {six_prob[2].avg:.4f}\t Situation_3 {six_prob[3].avg:.4f}\t Situation_4 {six_prob[4].avg:.4f}\t Situation_5 {six_prob[5].avg:.4f}\n\n'\
    .format(epoch, args.lr, Accuracys=accs, Loss=cls_loss, six_acc=six_accuracys, six_prob=six_probilities, acc_glg_value=acc_glg_value)

    with open(args.OutputPath + "/test_result_source.log","a") as f:
        f.writelines(LoggerInfo)

    #@ =========================================================================================================

    # Test on Target Domain
    acc, prec, recall = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    cls_loss, loss, afn_loss, data_time, batch_time = [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities, six_accuracys = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]
    acc_glg, prec_glg, recall_glg = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)] # sum prediction from global and global with local

    end = time.time()
    test_target_bar = tqdm(test_target_dataloader)
    for batch_index, (_, input, landmark, target) in enumerate(test_target_bar):
        data_time.update(time.time() - end)

        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()

        # Forward Propagation
        with torch.no_grad():
            end = time.time()
            #! m21-11-12，注释
            # feature, output, loc_output = model(input, landmark)
            #@ m21-11-11, 新的model调用
            features, preds = model(input, landmark)
            batch_time.update(time.time() - end)

        #@ m21-11-12, 新的计算loss的方式
        loss_ = 0
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i], target)
            cls_loss[i].update(tmp.cpu().data.item(), input.size(0))
            loss_ += tmp

        # Compute accuracy, precision and recall
        for classifier_id in range(7): # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id], target, acc[classifier_id], prec[classifier_id], recall[classifier_id])
        Compute_Accuracy(args, preds[0] + preds[6], target, acc_glg, prec_glg, recall_glg)


        Count_Probility_Accuracy(six_probilities, six_accuracys, preds, target)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        end = time.time()
        test_target_bar.desc = f'[Test (Target Domain) Epoch {epoch}/{args.epochs}] cls loss: {cls_loss[0].avg:.3f}, {cls_loss[1].avg:.3f}, {cls_loss[2].avg:.3f}, {cls_loss[3].avg:.3f}, {cls_loss[4].avg:.3f}, {cls_loss[5].avg:.3f}, {cls_loss[6].avg:.3f}'
        
    #! m21-11-12, 注释
    # AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    #@ m21-11-12, 只记录七个分类器的准确率
    accs = Show_OnlyAccuracy(acc)

    #@ m21-11-30, 计算glg的acc
    acc_glg_value = 0
    for class_id in range(7):
        acc_glg_value += acc_glg[class_id].avg
    acc_glg_value /= 7

    writer.add_scalars('Test/CLS_Acc_Target', {'global': accs[0], 'left_eye':accs[1], 'right_eye':accs[2], 'nose':accs[3], 'left_mouse':accs[4], 'right_mouse':accs[5], 'global_local':accs[6], 'global_local_global':acc_glg_value}, epoch)
    writer.add_scalars('Test/CLS_Loss_Target', {'global': cls_loss[0].avg, 'left_eye':cls_loss[1].avg, 'right_eye':cls_loss[2].avg, 'nose':cls_loss[3].avg, 'left_mouse':cls_loss[4].avg, 'right_mouse':cls_loss[5].avg, 'global_local':cls_loss[6].avg}, epoch)
    writer.add_scalars('Test/Six_Probilities_Target', {'situation_0': six_probilities[0].avg, 'situation_1':six_probilities[1].avg, 'situation_2':six_probilities[2].avg, 'situation_3':six_probilities[3].avg, 'situation_4':six_probilities[4].avg, 'situation_5':six_probilities[5].avg}, epoch)
    writer.add_scalars('Test/Six_Accuracys_Target', {'situation_0': six_accuracys[0].avg, 'situation_1':six_accuracys[1].avg, 'situation_2':six_accuracys[2].avg, 'situation_3':six_accuracys[3].avg, 'situation_4':six_accuracys[4].avg, 'situation_5':six_accuracys[5].avg}, epoch)


    LoggerInfo = '[Test (Target Domain) Epoch {0}]： Learning Rate {1} \n'\
    'Accuracy: global {Accuracys[0]:.4f}\t left_eye {Accuracys[1]:.4f}\t right_eye {Accuracys[2]:.4f}\t nose {Accuracys[3]:.4f}\t left_mouse {Accuracys[4]:.4f}\t right_mouse {Accuracys[5]:.4f}\t global_local {Accuracys[6]:.4f} global_local_global {acc_glg_value:.4f}\n'\
    'Cls_Loss: global {Loss[0].avg:.4f}\t left_eye {Loss[1].avg:.4f}\t right_eye {Loss[2].avg:.4f}\t nose {Loss[3].avg:.4f}\t left_mouse {Loss[4].avg:.4f}\t right_mouse {Loss[5].avg:.4f}\t global_local {Loss[6].avg:.4f}\n'\
    'Situ_Acc: Situation_0 {six_acc[0].avg:.4f}\t Situation_1 {six_acc[1].avg:.4f}\t Situation_2 {six_acc[2].avg:.4f}\t Situation_3 {six_acc[3].avg:.4f}\t Situation_4 {six_acc[4].avg:.4f}\t Situation_5 {six_acc[5].avg:.4f}\n'\
    'Situ_Pro: Situation_0 {six_prob[0].avg:.4f}\t Situation_1 {six_prob[1].avg:.4f}\t Situation_2 {six_prob[2].avg:.4f}\t Situation_3 {six_prob[3].avg:.4f}\t Situation_4 {six_prob[4].avg:.4f}\t Situation_5 {six_prob[5].avg:.4f}\n \n'\
    .format(epoch, args.lr, Accuracys=accs, Loss=cls_loss, six_acc=six_accuracys, six_prob=six_probilities, acc_glg_value=acc_glg_value)

    with open(args.OutputPath + "/test_result_target.log","a") as f:
        f.writelines(LoggerInfo)

    #@ Save Checkpoints
    classifier_name = {0:'global', 1:'left_eye', 2:'right_eye', 3:'nose', 4:'left_mouth', 5:'right_mouth', 6:'global_local', 7:'global_local_global'}
    accs.append(acc_glg_value)
    best_classifier_id = accs.index(max(accs))
    best_classifier = classifier_name[best_classifier_id]
    best_acc = accs[best_classifier_id]
    if best_acc > Best_Acc: # 根据Source Domain的效果判断是否存储
        Best_Acc = best_acc
        print("***************")
        print(f'[Save] Best Acc: {Best_Acc:.4f}, the classifier is {best_classifier}. Save the checkpoint!')
        print("***************")

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))
        else:
            torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))

    return Best_Acc

def main():
    """Main."""

    # Parse Argument
    # args = parser.parse_args()   # 构造参数
    args = construct_args()
    torch.manual_seed(args.seed) # 人工种子
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

    if args.isTest:
        print('Test Model.')
    else:
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Train Epoch: %d' % args.epochs)
        print('Weight Decay: %f' % args.weight_decay)

        if args.useAFN:
            print('Use AFN Loss: %s' % args.methodOfAFN)
            if args.methodOfAFN == 'HAFN':
                print('Radius of HAFN Loss: %f' % args.radius)
            else:
                print('Delta Radius of SAFN Loss: %f' % args.deltaRadius)
            print('Weight L2 nrom of AFN Loss: %f' % args.weight_L2norm)

        if args.useDAN:
            print('Use DAN Loss: %s' % args.methodOfDAN)
            print('Learning Rate(Adversarial Network): %f' % args.lr_ad)

    print('================================================')

    print('Number of classes : %d' % args.class_num)
    if not args.useLocalFeature:
        print('Only use global feature.')
    else:
        print('Use global feature and local feature.')

        if args.useIntraGCN:
            print('Use Intra GCN.')
        if args.useInterGCN:
            print('Use Inter GCN.')

        if args.useRandomMatrix and args.useAllOneMatrix:
            print('Wrong : Use RandomMatrix and AllOneMatrix both!')
            return None
        elif args.useRandomMatrix:
            print('Use Random Matrix in GCN.')
        elif args.useAllOneMatrix:
            print('Use All One Matrix in GCN.')

        if args.useCov and args.useCluster:
            print('Wrong : Use Cov and Cluster both!')
            return None
        else:
            if args.useCov:
                print('Use Mean and Cov.')
            else:
                print('Use Mean.') if not args.useCluster else print('Use Mean in Cluster.')

    print('================================================')


    print('================================================')
    # Bulid Model
    print('Building Model...')
    model = BulidModel(args)
    # with open(args.OutputPath + "/second.log", "a") as f:
    #     num = 0
    #     for k, v in model.named_parameters():
    #         num += 1
    #         f.writelines(str(num) + "、" + str(k) + "\n" + str(v) + "\n\n")
    print('Done!')
    print('================================================')

    # Bulid Dataloder
    print("Building Train and Test Dataloader...")
    train_source_dataloader, _ = BulidDataloader(args, flag1='train', flag2='source')
    train_target_dataloader, init_train_dataset_data = BulidDataloader(args, flag1='train', flag2='target')
    test_source_dataloader, _ = BulidDataloader(args, flag1='test', flag2='source')
    test_target_dataloader, _ = BulidDataloader(args, flag1='test', flag2='target')
    labeled_train_target_loader = None
    print('Done!')
    print('================================================')

    # Bulid Adversarial Network
    print('Building Adversarial Network...')
    #@ m21-11-13, 新写的对抗网络数组
    random_layers, ad_nets = [], []
    for i in range(2):
        random_layer, ad_net = BulidAdversarialNetwork(args, 64, args.class_num) if args.useDAN else (None, None)
        ad_nets.append(ad_net)
        random_layers.append(random_layer)
    random_layer, ad_net = BulidAdversarialNetwork(args, 384, args.class_num) if args.useDAN else (None, None)
    ad_nets.append(ad_net)
    random_layers.append(random_layer)
    print('Done!')
    #! m21-11-13, 注释
    # random_layer, ad_net = BulidAdversarialNetwork(args, model.output_num(), args.class_num) if args.useDAN else (None, None)
    print('================================================')


    # Set Optimizer #@ m21-11-13, 新增7个判别器的优化
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(args, model)
    optimizer = Set_Optimizer(args, param_optim, args.lr, args.weight_decay, args.momentum)

    param_optim_ads = []
    for i in range(len(ad_nets)):
        param_optim_ads += Set_Param_Optim(args, ad_nets[i]) if args.useDAN else None
    optimizer_ad = Set_Optimizer(args, param_optim_ads, args.lr_ad, args.weight_decay,
                                 args.momentum) if args.useDAN else None
    print('Done!')

    print('================================================')

    # Init Mean #! m21-11-13, 注释
    '''
        if args.useLocalFeature and not args.isTest:

            if args.useCov:
                print('Init Mean and Cov...')
                Initialize_Mean_Cov(args, model, False)
            else:
                if args.useCluster:
                    print('Initialize Mean in Cluster....')
                    Initialize_Mean_Cluster(args, model, False, train_source_dataloader, train_target_dataloader)
                else:
                    print('Init Mean...')
                    Initialize_Mean(args, model, False)

            torch.cuda.empty_cache()

            print('Done!')
            print('================================================')
        '''

    # Save Best Checkpoint
    Best_Accuracy = 0
    confidence, probility = 0, 0

    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter(os.path.join(args.OutputPath, 'visual_board'))

    for epoch in range(1, args.epochs + 1):

        if args.showFeature and epoch % 5 == 1:
            print(f"=================\ndraw the tSNE graph...")
            Visualization(args.OutputPath + '/result_pics/train/source/{}_Source.jpg'.format(epoch), model, dataloader=train_source_dataloader, useClassify=False, domain='Source')
            Visualization(args.OutputPath + '/result_pics/train/target/{}_Target.jpg'.format(epoch), model, train_target_dataloader, useClassify=False, domain='Target')

            VisualizationForTwoDomain(args.OutputPath + '/result_pics/train_tow_domain/{}_train'.format(epoch), model, train_source_dataloader, train_target_dataloader, useClassify=False, showClusterCenter=False)
            VisualizationForTwoDomain(args.OutputPath + '/result_pics/test_tow_domain/{}_test'.format(epoch), model, test_source_dataloader, test_target_dataloader, useClassify=False, showClusterCenter=False)
            print(f"finish drawing!\n=================")

        if not args.isTest:
            if args.useCluster and epoch % 10 == 0:
                print(f"=================\nupdate the running_mean...")
                Initialize_Mean_Cluster(args, model, False, train_source_dataloader, train_target_dataloader)
                torch.cuda.empty_cache()
                print(f"finish the updating!\n=================")
    
            labeled_train_target_loader_, confidence_, probility_ = BuildLabeledDataloader(args, train_target_dataloader, init_train_dataset_data, model)

            if confidence_ >= 0.99:
                labeled_train_target_loader = labeled_train_target_loader_
                confidence, probility = confidence_, probility_

            writer.add_scalars('Labeled_Prob_Confi', {'confidence': confidence, 'probility': probility}, epoch)

            Train(args, model, ad_nets, random_layers, train_source_dataloader, train_target_dataloader, labeled_train_target_loader, optimizer, optimizer_ad, epoch, writer)

        Best_Accuracy = Test(args, model, test_source_dataloader, test_target_dataloader, Best_Accuracy, epoch, writer)

    writer.close()
    print(f"==========================\n{args.Log_Name} is done, ")
    print(f"saved in：{args.OutputPath}")

if __name__ == '__main__':
    main()
