import os
import torch.utils.data as data
import numpy as np
import torch
import cv2
import skimage.io as skio


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class WDCD_train(data.Dataset):

    def __init__(self, root, image_set='train', input_form='TIFF', useMFC=False):

        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.in_form = input_form
        self.MFC = useMFC

        voc_root = self.root

        if self.in_form =='TIFF':
            image_dir = os.path.join(voc_root, 'JPEGImages')
        elif self.in_form =='RGB':
            image_dir = os.path.join(voc_root, 'JPEGImages_vis')

        if self.MFC:
            block_dir = os.path.join(voc_root, 'block_label_train')
        else:
            block_dir = os.path.join(voc_root, 'block_label_train')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        if self.in_form =='TIFF':
            self.images = [os.path.join(image_dir, x + ".tiff") for x in file_names]
        elif self.in_form =='RGB':
            self.images = [os.path.join(image_dir, x + ".png") for x in file_names]


        self.blocks = [os.path.join(block_dir, x + "_bl.npy") for x in file_names]

    # just return the img and target in P[key]
    def __getitem__(self, index):

        # get the hyperspectral data and lables
        if self.in_form =='TIFF':
            rsData = skio.imread(self.images[index],plugin="tifffile")
            rsData = np.asarray(rsData,dtype="float32")
            rsData = rsData.transpose(2, 0, 1)
        elif self.in_form == 'RGB':
            rsData = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
            rsData = np.asarray(rsData)
            rsData = rsData.transpose(2, 0, 1)

        img = torch.tensor(rsData)

        mean = torch.tensor([441.14345306, 443.68343218, 380.8605016, 307.75192367])
        mean_re = mean.view(4, 1, 1).expand((4, 250, 250))
        img = img - mean_re


        block_label = np.load(self.blocks[index])
        block_label = np.expand_dims(block_label, axis=0)

        return img, block_label

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)

class WDCD_test_cls(data.Dataset):
    def __init__(self, root, image_set='test',input_form='TIFF'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.in_form = input_form
        voc_root = self.root

        if self.image_set == 'train':
            image_dir = os.path.join(voc_root, 'JPEGImages')
            block_dir = os.path.join(voc_root, 'block_label_train')
        elif self.image_set == 'test':
            image_dir = os.path.join(voc_root, 'testing/image_cut')
            block_dir = os.path.join(voc_root, 'block_label_test')

        if not os.path.exists(image_dir):
            raise ValueError(
                'Wrong image_dir entered!')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/')
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

        self.blocks = [os.path.join(block_dir, x + "_bl.npy") for x in file_names]
        assert (len(self.images) == len(self.blocks))

    # just return the img and target in P[key]
    def __getitem__(self, index):
        # get the hyperspectral data and lables
        if self.in_form =='TIFF':
            rsData = np.load(self.images[index])
            rsData = np.asarray(rsData,dtype='float32')
            rsData = rsData.transpose(2, 0, 1)
        elif self.in_form == 'RGB':
            rsData = np.load(self.images[index])

        img = torch.Tensor(rsData)
        # test mean
        # mean = torch.tensor([301.37868060, 289.88378237, 259.03378350, 295.58795369])
        # train mean
        mean = torch.tensor([441.14345306, 443.68343218, 380.8605016, 307.75192367])
        # mean_re = mean.view(4, 1, 1).permute(1, 2, 0)
        mean_re = mean.view(4, 1, 1).expand((4, 250, 250))

        img = img - mean_re

        if self.transform is not None:
            hed = torch.zeros((250, 250))
            target = torch.zeros((250, 250))
            img, weak, hed = self.transform(img, target, hed)

        block_label = np.load(self.blocks[index])
        # block_label = np.expand_dims(block_label, axis=0)
        return img, block_label

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)

class WDCD_test_seg(data.Dataset):

    def __init__(self, root, image_set='train',input_form='TIFF', transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set
        self.in_form = input_form

        voc_root = self.root

        if self.in_form == 'TIFF':
            image_dir = os.path.join(voc_root, 'testing/image_cut')
        elif self.in_form == 'RGB':
            image_dir = os.path.join(voc_root, 'JPEGImages_vis')

        mask_dir = os.path.join(voc_root, 'testing/label_cut')
        block_dir= os.path.join(voc_root, 'block_label_test')

        if not os.path.exists(image_dir):
            raise ValueError(
                'Wrong image_dir entered!')


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
        assert (len(self.images) == len(self.masks) == len(self.blocks))

    # just return the img and target in P[key]
    def __getitem__(self, index):
        # get the hyperspectral data and lables
        if self.in_form =='TIFF':
            rsData = np.load(self.images[index])
            rsData = np.asarray(rsData,dtype='float32')
            rsData = rsData.transpose(2, 0, 1)
        elif self.in_form == 'RGB':
            rsData = np.load(self.images[index])

        target = np.load(self.masks[index])
        target = np.expand_dims(target, axis=0)

        img = torch.Tensor(rsData)

        # test mean
        # mean = torch.tensor([301.37868060, 289.88378237, 259.03378350, 295.58795369])
        # train mean
        mean = torch.tensor([441.14345306, 443.68343218, 380.8605016, 307.75192367])
        # mean_re = mean.view(4, 1, 1).permute(1, 2, 0)
        mean_re = mean.view(4, 1, 1).expand((4, 250, 250))

        img = img - mean_re

        target = torch.Tensor(target)


        block_label = np.load(self.blocks[index])
        block_label = np.expand_dims(block_label, axis=0)

        return img, target, block_label

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)


class WDCD_validate_seg(data.Dataset):

    def __init__(self, root, image_set='trainval',input_form='TIFF'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.in_form = input_form

        voc_root = self.root

        if self.in_form == 'TIFF':
            image_dir = os.path.join(voc_root, 'validation/image_cut')
        elif self.in_form == 'RGB':
            image_dir = os.path.join(voc_root, 'JPEGImages_vis')

        mask_dir = os.path.join(voc_root, 'validation/label_cut')
        block_dir= os.path.join(voc_root, 'validation/block_label_validate')

        if not os.path.exists(image_dir):
            raise ValueError(
                'Wrong image_dir entered!')


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
        assert (len(self.images) == len(self.masks) == len(self.blocks))

    # just return the img and target in P[key]
    def __getitem__(self, index):
        # get the hyperspectral data and lables
        if self.in_form =='TIFF':
            rsData = np.load(self.images[index])
            rsData = np.asarray(rsData,dtype='float32')
            rsData = rsData.transpose(2, 0, 1)
        elif self.in_form == 'RGB':
            rsData = np.load(self.images[index])

        target = np.load(self.masks[index])
        target = np.expand_dims(target, axis=0)

        img = torch.Tensor(rsData)

        # trainval mean
        mean = torch.tensor([499.55886780, 520.36863674, 459.07197668, 431.99547925])
        # mean_re = mean.view(4, 1, 1).permute(1, 2, 0)
        mean_re = mean.view(4, 1, 1).expand((4, 250, 250))

        img = img - mean_re
        target = torch.Tensor(target)

        block_label = np.load(self.blocks[index])
        block_label = np.expand_dims(block_label, axis=0)

        return img, target, block_label

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)
