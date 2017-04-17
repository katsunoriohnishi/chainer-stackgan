#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
from chainer import cuda
from chainer import Variable

def extension(data_processor, models, test_indices, args, data_type, rows=5, cols=5, seed=0):
    @chainer.training.make_extension()
    def make_video(trainer):
        gen_s1, gen_s2 = models
        
        ###[reconstract image from z]###
        xp = np if args.gpu < 0 else cuda.cupy
        np.random.seed(seed)
        x_in, c_in = data_processor.get_example4test(test_indices)
        c_in = Variable(xp.asarray(c_in))
        z = Variable(xp.asarray(gen_s1.make_hidden(rows * cols)))
        np.random.seed()
        
        x_lr_fake = gen_s1(z, c_in, test=True)
        y = gen_s2(x_lr_fake, c_in, test=True)

        if args.gpu>=0:
            y = chainer.cuda.to_cpu(y.data)

        y = np.asarray(np.clip((y + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = y.shape
        y = y.reshape((rows, cols, 3, H, W))
        y = y.transpose(0, 3, 1, 4, 2)
        y = y.reshape((rows * H, cols * W, 3))

        raw_img = np.asarray(np.clip((x_in + 1.) * (255. / 2.), 0.0, 255.0), np.uint8)
        _, _, H, W = raw_img.shape
        raw_img = raw_img.reshape((rows, cols, 3, H, W))
        raw_img = raw_img.transpose(0, 3, 1, 4, 2)
        raw_img = raw_img.reshape((rows * H, cols * W, 3))

        ###todo reshape raw_img, and hstack with generated images 
        preview_dir = '{}/preview/'.format(args.out) + '/{}/'.format(data_type)
        preview_path = preview_dir + \
                       '{:0>3}.jpg'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(np.hstack((raw_img, y))).save(preview_path)

    return make_video

def extension4test(data_processor, models, test_indices, args, data_type, rows=5, cols=5, seed=0):
    gen_s1, gen_s2 = models
    ###[reconstract image from z]###
    xp = np if args.gpu < 0 else cuda.cupy

    np.random.seed(seed)
    c_in = data_processor.get_example(test_indices)
    c_in = Variable(xp.asarray(c_in))
    z = Variable(xp.asarray(gen_s1.make_hidden(rows * cols)))
    np.random.seed()

    x_lr_fake = gen_s1(z, c_in, test=True)
    y = gen_s2(x_lr_fake, c_in, test=True)

    if args.gpu>=0:
        y = chainer.cuda.to_cpu(y.data)

    y = np.asarray(np.clip((y + 1.) * (255. / 2.), 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = y.shape
    y = y.reshape((rows, cols, 3, H, W))
    y = y.transpose(0, 3, 1, 4, 2)
    y = y.reshape((rows * H, cols * W, 3))

    ###todo reshape raw_img, and hstack with generated images 
    preview_dir = '{}/'.format(args.out) 
    preview_path = preview_dir + '{}_stage2.jpg'.format(data_type)
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(y).save(preview_path)

