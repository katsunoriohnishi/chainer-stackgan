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


from visualizer import extension
import net_stage2 as net_s2
from updater import Stage2_Updater
import pickle

import sys

sys.path.append('../misc')
from data_processor import PreprocessedDataset
import datetime

sys.path.append('../stage1')
import net_stage1 as net_s1
sys.path.pop()

# Setup optimizer
def make_optimizer(model, alpha=2e-4, beta1=0.5, beta2=0.999, epsilon=1e-8):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2, eps=epsilon)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
    return optimizer


def stage2_training(args, train, test):
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    gen_s1 = net_s1.Generator((64,64))
    serializers.load_npz(args.stage1_weight, gen_s1)

    # Prepare GAN model, defined in net.py
    gen_s2 = net_s2.Generator()
    dis_s2 = net_s2.Discriminator()
    
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen_s1.to_gpu()
        gen_s2.to_gpu()
        dis_s2.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    opt_gen = make_optimizer(gen_s2)
    opt_dis = make_optimizer(dis_s2)

    # Updater
    updater = Stage2_Updater(
        models=(gen_s1, gen_s2, dis_s2),
        iterator=train_iter,
        optimizer={'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu, KL_COEFF=2.0)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval), 'epoch'
    visualize_interval = (args.visualize_interval), 'epoch'
    log_interval = (args.log_interval), 'epoch'

    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)

    trainer.extend(extensions.LogReport(trigger=log_interval))

    trainer.extend(
        extensions.PlotReport(['gen/loss', 'dis/loss', 'gen/kl_loss'], trigger=visualize_interval,
                              file_name='plot.jpg'))

    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss', 'gen/kl_loss'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen_s2, 'gen_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis_s2, 'dis_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)

    np.random.seed(0)
    train_indices = np.random.randint(0, len(train), 25).tolist()
    test_indices = np.random.randint(0, len(test), 25).tolist()
    np.random.seed()

    trainer.extend(extension(train, (gen_s1, gen_s2), train_indices, args, 'train'), trigger=visualize_interval)
    trainer.extend(extension(test, (gen_s1, gen_s2), test_indices, args, 'test'), trigger=visualize_interval)

    if args.adam_decay_epoch:
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt_gen),
                       trigger=(args.adam_decay_epoch, 'epoch'))
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt_dis),
                       trigger=(args.adam_decay_epoch, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


def main():
    todaydetail = datetime.datetime.today()
    parser = argparse.ArgumentParser(description='Chainer example: StackGAN')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=600, type=int,
                        help='number of epochs (default: 600)')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='learning minibatch size (default: 64)')
    parser.add_argument('--log_interval', '-li', type=int, default=1,
                        help='log interval epoch (default: 1)')
    parser.add_argument('--visualize_interval', '-vi', type=int, default=10,
                        help='visualize interval epoch (default: 10)')
    parser.add_argument('--snapshot_interval', '-si', type=int, default=100,
                        help='snapshot interval epoch (default: 100)')
    parser.add_argument('--out', '-o', default='../result/stage2/' + todaydetail.strftime("%m%d_%H%M%S") + '/',
                        help='Output directory')
    parser.add_argument('--data_dir', '-d', default='../data',
                        help='data root directory')
    parser.add_argument('--adam_decay_epoch', '-ade', type=int, default=100,
                        help='adam decay epoch (default: 100)')
    parser.add_argument('--stage1_weight', '-s1w',
                        help='path to stage1_weight')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    ## dataset
    print('loading bird dataset')
    img_train_path = args.data_dir + '/birds/train/304images.pickle'
    with open(img_train_path, 'rb') as f_in:
        x_train = pickle.load(f_in)
    text_train_path = args.data_dir + '/birds/train/char-CNN-RNN-embeddings.pickle'
    with open(text_train_path, 'rb') as f_in:
        c_train = pickle.load(f_in)

    img_test_path = args.data_dir + '/birds/test/304images.pickle'
    with open(img_test_path, 'rb') as f_in:
        x_test = pickle.load(f_in)
    text_test_path = args.data_dir + '/birds/test/char-CNN-RNN-embeddings.pickle'
    with open(text_test_path, 'rb') as f_in:
        c_test = pickle.load(f_in)

    train = PreprocessedDataset(x_train, c_train, crop_size=(256,256))
    test = PreprocessedDataset(x_test, c_test, crop_size=(256,256))

    ## main testing
    stage2_training(args, train, test)

if __name__ == '__main__':
    main()
