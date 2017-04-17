#!/usr/bin/env python

from __future__ import print_function

import os

import numpy as np
from PIL import Image

import chainer
from chainer import cuda
from chainer import Variable

def extension(data_processor, model, test_indices, args, data_type, rows=5, cols=5, seed=0):
    @chainer.training.make_extension()
    def make_video(trainer):            
        ### reconstruct image from z###
        ## prepare input data
        xp = np if args.gpu < 0 else cuda.cupy
        np.random.seed(seed)
        x_in, c_in = data_processor.get_example4test(test_indices)
        c_in = Variable(xp.asarray(c_in))
        z = Variable(xp.asarray(model.make_hidden(rows*cols)))
        np.random.seed()
        
        ## generate
        y = model(z, c_in, test=True)

        ### save generated images
        ## post process generated images
        if args.gpu>=0:
            y = chainer.cuda.to_cpu(y.data)

        y = np.asarray(np.clip((y+1.)*(255./2.), 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = y.shape
        y = y.reshape((rows, cols, 3, H, W))
        y = y.transpose(0, 3, 1, 4, 2)
        y = y.reshape((rows * H, cols * W, 3))

        ## post process real images in the same way as generated images
        raw_img = np.asarray(np.clip((x_in+1.)*(255./2.), 0.0, 255.0), np.uint8)
        _, _, H, W = raw_img.shape
        raw_img = raw_img.reshape((rows, cols, 3, H, W))
        raw_img = raw_img.transpose(0, 3, 1, 4, 2)
        raw_img = raw_img.reshape((rows * H, cols * W, 3))

        ### save images
        preview_dir = '{}/preview/'.format(args.out) + '/{}/'.format(data_type)
        preview_path = preview_dir +\
                       '{:0>3}.jpg'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(np.hstack((raw_img, y))).save(preview_path)

    return make_video
