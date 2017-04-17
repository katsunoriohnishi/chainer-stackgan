#!/usr/bin/env python

from __future__ import print_function

import argparse

import matplotlib

# Disable interactive backend
matplotlib.use('Agg')
import numpy as np

import chainer
from chainer import cuda
from chainer.training import extensions
from chainer import training
from chainer import serializers

from visualizer import extension4test
import net_stage2 as net_s2

import pickle
import datetime
import sys

sys.path.append('../misc')
from data_processor import PreprocessedDataset4Test

sys.path.append('../stage1')
import net_stage1 as net_s1

def stage2_test(args, test):
    gen_s1 = net_s1.Generator((64, 64))
    serializers.load_npz(args.stage1_weight, gen_s1)

    gen_s2 = net_s2.Generator()
    serializers.load_npz(args.stage2_weight, gen_s2)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen_s1.to_gpu()
        gen_s2.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    np.random.seed(0)
    test_indices = np.random.randint(0, len(test), 25).tolist()
    np.random.seed()

    extension4test(test, (gen_s1, gen_s2), test_indices, args, args.data_type)

def main():
    todaydetail = datetime.datetime.today()
    parser = argparse.ArgumentParser(description='Chainer example: StackGAN')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='../result/demo/',
                        help='Output directory')
    parser.add_argument('--data_dir', '-dd', default='../data',
                        help='data root directory')
    parser.add_argument('--data_type', '-dt', default='test',
                        help='')
    parser.add_argument('--stage1_weight', '-s1w', default='../models/stage1_gen_600.npz',
                        help='path to stage1_weight')
    parser.add_argument('--stage2_weight', '-s2w', default='../models/stage2_gen_600.npz',
                        help='path to stage2_weight')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('')

    ## dataset
    print('loading bird dataset')
    text_test_path = args.data_dir + '/cub/test/char-CNN-RNN-embeddings.pickle'
    with open(text_test_path, 'rb') as f_in:
        c_test = pickle.load(f_in)

    test = PreprocessedDataset4Test(c_test)

    ## main testing
    stage2_test(args, test)

if __name__ == '__main__':
    main()
