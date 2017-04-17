#!/usr/bin/env python

from __future__ import print_function

import chainer
import random
from PIL import Image
import numpy as np
import scipy.misc

class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, img_dataset, text_dataset, crop_size=(64,64)):
        self.img_dataset = img_dataset
        self.text_dataset = text_dataset
        self.crop_x = crop_size[0]
        self.crop_y = crop_size[1]

    def __len__(self):
        return len(self.img_dataset)

    def preprocess_img(self, image, flipcrop=True):
        crop_x = self.crop_x
        crop_y = self.crop_y
        _, h, w = image.shape
        
        if flipcrop:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_y - 1)
            left = random.randint(0, w - crop_x - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
            bottom = top + crop_y
            right = left + crop_x
            image = image[:, top:bottom, left:right]
        else:
            image = scipy.misc.imresize(image.transpose(1,2,0), [self.crop_y, self.crop_x], 'bicubic').transpose(2,0,1)

        image_src = image[:]
        image = image.astype(np.float32) * (2 / 255.) - 1.
        return image, image_src

    def get_example(self, i, train=True):
        img = self.img_dataset[i].transpose(2,0,1) ## (Y,X,ch) -> (ch,Y,X)
        img, img_src = self.preprocess_img(img, train)
        j = np.random.randint(10)
        text_feat = self.text_dataset[i][j,:]
        
        if train:
            return img, text_feat
        else:
            return img, text_feat, img_src
    
    def get_example4test(self, test_indices):
        Imgs = []
        Texts = []
        for i in test_indices:
            img, text, _ = self.get_example(i, train=False)
            Imgs.append(img)
            Texts.append(text)
        return np.asarray(Imgs).transpose(0,3,1,2), np.asarray(Texts)

class PreprocessedDataset4Test(chainer.dataset.DatasetMixin):
    def __init__(self, text_dataset):
        self.text_dataset = text_dataset

    def __len__(self):
        return len(self.text_dataset)

    def get_example(self, indices):
        Texts = []
        for i in indices:
            j = np.random.randint(10)
            text = self.text_dataset[i][j, :]
            Texts.append(text)
        return np.asarray(Texts)
