#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：L5 -> LFW_Dataset
@IDE    ：PyCharm
@Author ：DIPTE
@Date   ：2020/8/2 2:22
@Desc   ：
=================================================='''
import os, sys
import cv2
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import pandas as pd

class LFW(Dataset):
    def __init__(self, root_path, file_list, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.nameLs = []
        self.nameRs = []
        self.flags = []
        self.folds = []

        with open(file_list) as f:
            lines = f.readlines()
            for i, line in enumerate(lines[1:]):
                line = line.strip()
                p = line.split('\t')
                fold = i // 600  # 6000 pairs, so there should be 10 folds
                if len(p) == 3:  # same person
                    nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                    nameR = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
                    flag = 1
                elif len(p) == 4:
                    nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                    nameR = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
                    flag = -1
                self.nameLs.append(nameL)
                self.nameRs.append(nameR)
                self.flags.append(flag)
                self.folds.append(fold)

    def __getitem__(self, index):
        img_l = cv2.imread((os.path.join(self.root_path, self.nameLs[index])), 1)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r = cv2.imread((os.path.join(self.root_path, self.nameRs[index])), 1)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img_l = self.transform(img_l)
            img_r = self.transform(img_r)
        else:
            img_l = torch.from_numpy(img_l)
            img_r = torch.from_numpy(img_r)
        return {'pair': [img_l, img_r],
                'label': self.flags[index],
                'fold': self.folds[index]}

    def __len__(self):
        return len(self.flags)


class CASIAWebFace(Dataset):
    def __init__(self, root_path, file_list, transform=None):
        self.root_path = root_path
        self.transform = transform

        image_list = []
        label_list = []
        with open(file_list) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_path, label_name = line.split(' ')
                image_list.append(image_path)
                label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.num_class = len(np.unique(self.label_list))
        print("CASIA dataset size:", len(self.image_list), '/', self.num_class)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = self.label_list[index]

        image = cv2.imread(os.path.join(self.root_path, image_path), 1)  # Read BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        flip = np.random.choice(2) * 2 - 1
        if flip == 1:
            image = cv2.flip(image, 1)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)

        return image, label

    def __len__(self):
        return len(self.image_list)


class SiameseCASIAWebFace(Dataset):
    def __init__(self, root_path, file_list, transform=None):
        self.root_path = root_path
        self.transform = transform

        image_list = []
        label_list = []
        with open(file_list) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_path, label_name = line.split(' ')
                image_list.append(image_path)
                label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.label_set = set(self.label_list)
        self.num_class = len(self.label_set)
        self.label_array = np.array(label_list)
        self.label_to_idxs = {label: np.where(self.label_array == label)[0]
                              for label in self.label_set}
        print("CASIA dataset size:", len(self.image_list), '/', self.num_class)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        imageL_path, labelL = self.image_list[index], self.label_list[index]
        if target == 1:  #
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_idxs[labelL])

        else:
            siamese_label = np.random.choice(list(self.label_set - set([labelL])))
            siamese_index = np.random.choice(self.label_to_idxs[siamese_label])
        imageR_path = self.image_list[siamese_index]
        labelR = self.label_list[siamese_index]

        imageL = cv2.imread(os.path.join(self.root_path, imageL_path), 1)
        imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)
        imageR = cv2.imread(os.path.join(self.root_path, imageR_path), 1)
        imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB)

        flip = np.random.choice(2) * 2 - 1
        if flip == 1:
            imageL = cv2.flip(imageL, 1)
            imageR = cv2.flip(imageR, 1)
        if self.transform is not None:
            imageL = self.transform(imageL)
            imageR = self.transform(imageR)
        else:
            imageL = torch.from_numpy(imageL)
            imageR = torch.from_numpy(imageR)

        return imageL, imageR, target


class TripletFaceDataset(Dataset):
    # Modified to add 'training_triplets_path' parameter
    def __init__(self, root_dir, csv_name, num_triplets, training_triplets_path=None, transform=None):

        # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
        #   VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
        #   forcing the 'name' column as being of type 'int' instead of type 'object')
        self.df = pd.read_csv(csv_name, dtype={'id': object, 'name': object, 'class': int})
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.transform = transform
        # self.num_class = len(np.unique(self.label_list))
        self.num_class = len(self.df['class'].unique())#len(self.label_set)
        # Modified here
        if training_triplets_path is None:
            self.training_triplets = self.generate_triplets(self.df, self.num_triplets)
        else:
            self.training_triplets = np.load(training_triplets_path)

    @staticmethod
    def generate_triplets(df, num_triplets):

        def make_dictionary_for_face_class(df):
            """
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            """
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append(df.iloc[idx, 0])

            return face_classes

        triplets = []
        classes = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)

        # Modified here to add a print statement
        print("\nGenerating {} triplets...".format(num_triplets))

        progress_bar = tqdm(range(num_triplets))
        for _ in progress_bar:

            """
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            """

            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)

            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)

            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)

            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))

                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))

            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [
                    face_classes[pos_class][ianc],
                    face_classes[pos_class][ipos],
                    face_classes[neg_class][ineg],
                    pos_class,
                    neg_class,
                    pos_name,
                    neg_name
                ]
            )

        # Modified here to save the training triplets as a numpy file to not have to redo this process every
        #   training execution from scratch
        print("Saving training triplets list in datasets/ directory ...")
        np.save('datasets/training_triplets_{}.npy'.format(num_triplets), triplets)
        print("Training triplets' list Saved!\n")

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = self.add_extension(os.path.join(self.root_dir, str(pos_name), str(anc_id)))
        pos_img = self.add_extension(os.path.join(self.root_dir, str(pos_name), str(pos_id)))
        neg_img = self.add_extension(os.path.join(self.root_dir, str(neg_name), str(neg_id)))

        # Modified to open as PIL image in the first place
        # anc_img = Image.open(anc_img)
        # pos_img = Image.open(pos_img)
        # neg_img = Image.open(neg_img)

        anc_img = cv2.imread(anc_img, 1)  # Read BGR
        anc_img = cv2.cvtColor(anc_img, cv2.COLOR_BGR2RGB)
        pos_img = cv2.imread(pos_img, 1)  # Read BGR
        pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2RGB)
        neg_img = cv2.imread(neg_img, 1)  # Read BGR
        neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)


        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            # print('sample',type(sample['anc_img']))
            # print(sample['pos_img'])
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)

    # Added this method to allow .jpg and .png image support
    def add_extension(self, path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)