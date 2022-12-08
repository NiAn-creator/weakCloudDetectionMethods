import os
import torch.utils.data as data
import numpy as np
import torch
import cv2


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class Landsat_WDCD_train(data.Dataset):

    def __init__(self, root, image_set='train', input_form='TIFF',transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set
        self.in_form = input_form

        voc_root = self.root

        if self.in_form =='TIFF':
            image_dir = os.path.join(voc_root, 'JPEGImages')
        elif self.in_form =='RGB':
            image_dir = os.path.join(voc_root, 'JPEGImages_vis')

        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        block_dir = os.path.join(voc_root, 'block_label_train')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        if self.in_form =='TIFF':
            self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        elif self.in_form =='RGB':
            self.images = [os.path.join(image_dir, x + ".png") for x in file_names]

        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]
        self.blocks = [os.path.join(block_dir, x + "_bl.npy") for x in file_names]
        assert (len(self.images) == len(self.blocks))

    # just return the img and target in P[key]
    def __getitem__(self, index):

        # get the hyperspectral data and lables
        if self.in_form =='TIFF':
            rsData = np.load(self.images[index])
            rsData = np.asarray(rsData,dtype=np.float32)
        elif self.in_form == 'RGB':
            rsData = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
            rsData = np.asarray(rsData)
            rsData = rsData.transpose(2, 0, 1)

        img = torch.tensor(rsData[:, :-1, :-1])

        mean = torch.tensor([18224.1647880933961, 17301.0807253831552, 17627.28788760243, 20419.9637850012330])
        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        block_label = np.load(self.blocks[index])
        block_label = np.expand_dims(block_label, axis=0)
        block_label = torch.Tensor(block_label)

        mask = np.load(self.masks[index])
        target = torch.Tensor(mask[:-1, :-1])
        return img, block_label, target

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)

class Landsat_WDCD_valid(data.Dataset):

    def __init__(self, root, image_set='test',input_form='TIFF',transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set
        self.in_form = input_form

        voc_root = self.root

        if self.in_form == 'TIFF':
            image_dir = os.path.join(voc_root, 'JPEGImages')
        elif self.in_form == 'RGB':
            image_dir = os.path.join(voc_root, 'JPEGImages_vis')

        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        block_dir = os.path.join(voc_root, 'block_label_test')

        if not os.path.exists(image_dir):
            raise ValueError(
                'Wrong image_dir entered!')
        if not os.path.exists(mask_dir):
            raise ValueError(
                'Wrong mask_dir entered!')
        if not os.path.exists(block_dir):
            raise ValueError(
                'Wrong block_dir entered!')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        if self.in_form == 'TIFF':
            self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        elif self.in_form == 'RGB':
            self.images = [os.path.join(image_dir, x + ".png") for x in file_names]

        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]
        self.blocks = [os.path.join(block_dir, x + "_bl.npy") for x in file_names]
        assert (len(self.images) == len(self.blocks))

    # just return the img and target in P[key]
    def __getitem__(self, index):
        # get the hyperspectral data and lables

        if self.in_form =='TIFF':
            rsData = np.load(self.images[index])
            rsData = np.asarray(rsData,dtype=np.float32)
        elif self.in_form == 'RGB':
            rsData = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
            rsData = np.asarray(rsData)
            rsData = rsData.transpose(2, 0, 1)

        block_label = np.load(self.blocks[index])
        block_label = np.expand_dims(block_label, axis=0)

        img = torch.Tensor(rsData[:, :-1, :-1])

        mean = torch.tensor([18296.131356905466, 17319.720398339658, 17774.15391870007, 20126.780138424132])
        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        mask = np.load(self.masks[index])
        mask = np.asarray(mask,dtype=np.float32)
        target = torch.Tensor(mask[:-1, :-1])

        return img, block_label, target

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)