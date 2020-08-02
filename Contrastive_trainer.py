#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：L5 -> Contrastive_train
@IDE    ：PyCharm
@Author ：DIPTE
@Date   ：2020/8/3 0:25
@Desc   ：
=================================================='''
import time
import datetime

import torch
import torch.nn as nn
import numpy as np

from loss_function import *
from average_meter import AverageMeter

class Trainer(object):

    def __init__(self, epochs, dataloaders, model, optimizer, scheduler, device,
                 margin, ckpt_tag,criterion):

        self.epochs = epochs
        self.dataloaders = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.margin = margin
        self.ckpt_tag = ckpt_tag

        # save best model
        self.best_val_acc = -100


        self.criterion = criterion#ContrastiveLoss(margin=0.8)


    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch, 'train')
            self.eval_epoch(epoch, 'LFW')
        print("Best acc on LFW: {}, best threshold: {}".format(self.best_val_acc,
                                                               self.best_threshold))

    def train_epoch(self, epoch, phase):
        loss_ = AverageMeter()
        accuracy_ = AverageMeter()
        self.model.train()
        self.margin.train()

        for batch_idx, sample in enumerate(self.dataloaders[phase]):

            imageL, imageR, label = sample[0].to(self.device), \
                                    sample[1].to(self.device), sample[2].to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputL, outputR = self.model(imageL), self.model(imageR)

                acc = 0
                loss = self.criterion([outputL, outputR], label)

                loss.backward()
                self.optimizer.step()

            loss_.update(loss, label.size(0))
            accuracy_.update(acc, label.size(0))

            if batch_idx % 40 == 0:
                print('Train Epoch: {} [{:08d}/{:08d} ({:02.0f}%)]\tLoss:{:.6f}\tAcc:{:.6f} LR:{:.7f}'.format(
                    epoch, batch_idx * len(label), len(self.dataloaders[phase].dataset),
                           100. * batch_idx / len(self.dataloaders[phase]), loss.item(), 0,
                    self.optimizer.param_groups[0]['lr']))

        self.scheduler.step()

        print("Train Epoch Loss: {:.6f} Accuracy: {:.6f}".format(loss_.avg,
                                                                 accuracy_.avg))
        torch.save(self.model.state_dict(),
                   './checkpoints/{}_{}_Contrastive_{:04d}.pth'.format(self.ckpt_tag,
                                                           str(self.margin), epoch))
        torch.save(self.margin.state_dict(),
                   './checkpoints/{}_512_{}_Contrastive_{:04d}.pth'.format(self.ckpt_tag,
                                                               str(self.margin), epoch))

    def eval_epoch(self, epoch, phase):
        feature_ls = feature_rs = flags = folds = None
        # sample = {'pair':[img_l, img_r], 'label': 1/-1}
        for batch_idx, sample in enumerate(self.dataloaders[phase]):
            img_l = sample['pair'][0].to(self.device)
            img_r = sample['pair'][1].to(self.device)
            flag = sample['label'].numpy()
            fold = sample['fold'].numpy()
            feature_l, feature_r = self.getDeepFeature(img_l, img_r)
            feature_l, feature_r = feature_l.cpu().numpy(), feature_r.cpu().numpy()

            if (feature_ls is None) and (feature_rs is None):
                feature_ls = feature_l
                feature_rs = feature_r
                flags = flag
                folds = fold
            else:
                feature_ls = np.concatenate((feature_ls, feature_l), 0)
                feature_rs = np.concatenate((feature_rs, feature_r), 0)
                flags = np.concatenate((flags, flag), 0)
                folds = np.concatenate((folds, fold), 0)

        accs, thresholds = self.evaluation_10_fold(feature_ls, feature_rs, flags, folds,
                                                   method='cos_distance')

        print("Eval Epoch Average Acc: {:.4f}, Average Threshold: {:.4f}".format(
            np.mean(accs), np.mean(thresholds)))
        if np.mean(accs) > self.best_val_acc:
            self.best_val_acc = np.mean(accs)
            torch.save(self.model.state_dict(),
                       './checkpoints/{}_{}_Contrastive_best.pth'.format(self.ckpt_tag,
                                                             str(self.margin)))
            torch.save(self.margin.state_dict(),
                       './checkpoints/{}_512_{}_Contrastive_best.pth'.format(self.ckpt_tag, str(self.margin)))
            self.best_threshold = np.mean(thresholds)

    def getDeepFeature(self, img_l, img_r):
        self.model.eval()
        with torch.no_grad():
            feature_l = self.model(img_l)
            feature_r = self.model(img_r)
        return feature_l, feature_r

    def evaluation_10_fold(self, feature_ls, feature_rs, flags, folds,
                           method='l2_distance'):
        accs = np.zeros(10)
        thresholds = np.zeros(10)
        for i in range(10):
            val_fold = (folds != i)
            test_fold = (folds == i)
            # minus by mean
            mu = np.mean(np.concatenate((feature_ls[val_fold, :],
                                         feature_rs[val_fold, :]),
                                        0), 0)
            feature_ls = feature_ls - mu
            feature_rs = feature_rs - mu
            # normalization
            feature_ls = feature_ls / np.expand_dims(np.sqrt(np.sum(np.power(feature_ls, 2), 1)), 1)
            feature_rs = feature_rs / np.expand_dims(np.sqrt(np.sum(np.power(feature_rs, 2), 1)), 1)

            if method == 'l2_distance':
                scores = np.sum(np.power((feature_ls - feature_rs), 2), 1)
            elif method == 'cos_distance':
                scores = np.sum(np.multiply(feature_ls, feature_rs), 1)
            else:
                raise NameError("Distance Method not supported")
            thresholds[i] = self.getThreshold(scores[val_fold], flags[val_fold], 10000, method)
            accs[i] = self.getAccuracy(scores[test_fold], flags[test_fold], thresholds[i], method)

        return accs, thresholds

    def getThreshold(self, scores, flags, thrNum, method='l2_distance'):
        accs = np.zeros((2 * thrNum + 1, 1))
        thresholds = np.arange(-thrNum, thrNum + 1) * 3 / thrNum
        # print(thresholds)
        # print(np.min(scores))
        # print(np.max(scores))
        for i in range(2 * thrNum + 1):
            accs[i] = self.getAccuracy(scores, flags, thresholds[i], method)
        max_index = np.squeeze(accs == np.max(accs))
        best_threshold = np.mean(thresholds[max_index])  # multi best threshold
        return best_threshold

    def getAccuracy(self, scores, flags, threshold, method='l2_distance'):

        if method == 'l2_distance':
            pred_flags = np.where(scores < threshold, 1, -1)
        elif method == 'cos_distance':
            pred_flags = np.where(scores > threshold, 1, -1)

        acc = np.sum(pred_flags == flags) / pred_flags.shape[0]
        return acc

