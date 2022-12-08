import os
import torch.utils.data as data
import numpy as np
import torch
import cv2


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class WSFNet_cls_GF1(data.Dataset):

    def __init__(self, root, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')

        # mask_dir = os.path.join(voc_root, 'SegmentationClass')
        block_dir = os.path.join(voc_root, 'block_label_20')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        # self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]
        self.blocks = [os.path.join(block_dir, x + ".npy") for x in file_names]
        assert (len(self.images) == len(self.blocks))

    # just return the img and target in P[key]
    def __getitem__(self, index):

        # get the hyperspectral data and lables
        if len(self.images[index].split('_'))>9:
            rsData = np.load(self.images[index].replace('_1.npy','.npy'))
        else:
            rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)
        rsData = rsData.transpose(2, 0, 1)

        img = torch.tensor(rsData[:,:-1,:-1])

        if 'train' in self.image_set:
            mean = torch.tensor([493.98721723, 478.55947529, 424.43626955, 520.79453991])
        elif 'test' in self.image_set:
            mean = torch.tensor([458.23624615, 467.27988648, 422.11874812, 579.80099140])

        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        block_label = np.load(self.blocks[index])
        block_label = np.expand_dims(block_label, axis=0)
        block_label = torch.Tensor(block_label)

        return img, block_label

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)


class WSFNet_cls_GF1_vis(data.Dataset):

    def __init__(self, root, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]

    # just return the img and target in P[key]
    def __getitem__(self, index):

        # get the hyperspectral data and lables
        if len(self.images[index].split('_'))>9:
            rsData = np.load(self.images[index].replace('_1.npy','.npy'))
        else:
            rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)
        rsData = rsData.transpose(2, 0, 1)

        img = torch.tensor(rsData[:,:-1,:-1])

        if 'train' in self.image_set:
            mean = torch.tensor([493.98721723, 478.55947529, 424.43626955, 520.79453991])
        elif 'test' in self.image_set:
            mean = torch.tensor([458.23624615, 467.27988648, 422.11874812, 579.80099140])

        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        return img

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)
