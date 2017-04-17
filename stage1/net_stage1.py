#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

import sys
sys.path.append('../misc')
from custom_opt import CustomConv2D

class Generator(chainer.Chain):
    def __init__(self, img_size=(64,64), n_hidden=100, gf_dim=128, ef_dim=128):
        self.n_hidden = n_hidden
        self.s = img_size[0]
        self.s16 =  int(self.s // 16)
        self.gf_dim = gf_dim
        self.ef_dim = ef_dim
        w = chainer.initializers.Normal(0.02)
        super(Generator, self).__init__(
            ### for generation
            l0=L.Linear(None, self.s16**2 *self.gf_dim*8, initialW=w),
            bn0=L.BatchNormalization(self.s16 ** 2 * self.gf_dim * 8),
            
            dc1_1=CustomConv2D(None, gf_dim*2, 1, 1, 0, bn=True, activation=F.relu),
            dc1_2=CustomConv2D(None, gf_dim*2, 3, 1, 1, bn=True, activation=F.relu),
            dc1_3=CustomConv2D(None, gf_dim*8, 3, 1, 1, bn=True, activation=None),

            dc2=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=None, resize=True),

            dc2_1=CustomConv2D(None, gf_dim * 1, 1, 1, 0, bn=True, activation=F.relu),
            dc2_2=CustomConv2D(None, gf_dim * 1, 3, 1, 1, bn=True, activation=F.relu),
            dc2_3=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=None),

            dc3=CustomConv2D(None, gf_dim * 2, 3, 1, 1, bn=True, activation=F.relu, resize=True),
            dc4=CustomConv2D(None, gf_dim, 3, 1, 1, bn=True, activation=F.relu, resize=True),
            dc5=CustomConv2D(None, 3, 3, 1, 1, bn=False, activation=None, resize=True),

            ### for text condition augmentation
            lc_mu=L.Linear(None, self.ef_dim, initialW=w),
            lc_var=L.Linear(None, self.ef_dim, initialW=w),
        )

    def make_hidden(self, batchsize):
        return numpy.random.normal(0, 1, (batchsize, self.n_hidden))\
            .astype(numpy.float32)

    def __call__(self, z, c, test=False):
        ### text augmentation
        hc_mu = F.leaky_relu(self.lc_mu(c))
        hc_var = F.leaky_relu(self.lc_var(c))
        h_c = F.gaussian(hc_mu, hc_var)
        
        ### concate z and c
        h = F.concat((z, h_c))

        ### generate image
        h1_0 = F.reshape(self.bn0(self.l0(h), test=test), (h.data.shape[0], self.gf_dim*8, self.s16, self.s16))
        h1_1 = self.dc1_1(h1_0, test=test)
        h1_1 = self.dc1_2(h1_1, test=test)
        h1_1 = self.dc1_3(h1_1, test=test)
        h = F.relu(h1_0+h1_1)
        
        h2_0 = self.dc2(h, test=test)
        h2_1 = self.dc2_1(h2_0, test=test)
        h2_1 = self.dc2_2(h2_1, test=test)
        h2_1 = self.dc2_3(h2_1, test=test)  
        h = F.relu(h2_0+h2_1)
        
        h = self.dc3(h, test=test)
        h = self.dc4(h, test=test)

        x = F.tanh(self.dc5(h, test=test))
        if test:
            return x
        else:
            return x, hc_mu, hc_var

class Discriminator(chainer.Chain):

    def __init__(self, img_size=(64,64), df_dim=64, ef_dim=128):
        self.s = img_size[0]
        self.s16 = self.s//16
        self.df_dim=df_dim
        self.ef_dim=ef_dim
        w = chainer.initializers.Normal(0.02)
        super(Discriminator, self).__init__(
            ### for generated image
            c10_1=CustomConv2D(None, self.df_dim, 4, 2, 1, bn=False, activation=F.leaky_relu),
            c10_2=CustomConv2D(None, self.df_dim * 2, 4, 2, 1, bn=True, activation=F.leaky_relu),
            c10_3=CustomConv2D(None, self.df_dim * 4, 4, 2, 1, bn=True, activation=None),
            c10_4=CustomConv2D(None, self.df_dim * 8, 4, 2, 1, bn=True, activation=None),

            c11_1=CustomConv2D(None, self.df_dim * 2, 1, 1, 0, bn=True, activation=F.leaky_relu),
            c11_2=CustomConv2D(None, self.df_dim * 2, 3, 1, 1, bn=True, activation=F.leaky_relu),
            c11_3=CustomConv2D(None, self.df_dim * 8, 3, 1, 1, bn=True, activation=None),
            
            ### for text
            l_c = L.Linear(None, self.ef_dim, initialW=w),
            
            ### after concate
            c2 = CustomConv2D(None, self.df_dim * 8, 1, 1, 0, bn=True, activation=F.leaky_relu),
            l3 = L.Linear(None, 1, initialW=w)
        )

    def __call__(self, x, c, test=False):
        ### image discriminator
        h1_0 = self.c10_1(x, test=test)
        h1_0 = self.c10_2(h1_0, test=test)
        h1_0 = self.c10_3(h1_0, test=test)
        h1_0 = self.c10_4(h1_0, test=test)

        h1_1 = self.c11_1(h1_0, test=test)
        h1_1 = self.c11_2(h1_1, test=test)
        h1_1 = self.c11_3(h1_1, test=test)

        h = F.leaky_relu(h1_0+h1_1)

        ### text Compression and Spatial Replication
        h_c = F.leaky_relu(self.l_c(c))
        h_c = F.expand_dims(h_c, axis=2)
        h_c = F.expand_dims(h_c, axis=2)
        h_c = F.tile(h_c, (1, 1, self.s16, self.s16))

        h = F.concat((h, h_c))

        ### after concatenate
        h = self.c2(h, test=test)
        return self.l3(h)

class GeneratorSimple(chainer.Chain):
    def __init__(self, img_size=(64,64), n_hidden=100, gf_dim=128, ef_dim=128):
        self.s = img_size[0]
        self.s16 = int(self.s // 16)
        self.gf_dim = gf_dim
        self.ef_dim = ef_dim
        self.n_hidden=n_hidden
        w = chainer.initializers.Normal(0.02)
        super(GeneratorSimple, self).__init__(
            ### for image
            l0=L.Linear(None, self.s16 ** 2 * self.gf_dim * 8, initialW=w),
            dc1=L.Deconvolution2D(None, self.gf_dim * 4, 4, 2, 1, initialW=w),
            dc2=L.Deconvolution2D(None, self.gf_dim * 2, 4, 2, 1, initialW=w),
            dc3=L.Deconvolution2D(None, self.gf_dim, 4, 2, 1, initialW=w),
            dc4=L.Deconvolution2D(None, 3, 4, 2, 1, initialW=w),
            bn0=L.BatchNormalization(self.s16 ** 2 * self.gf_dim * 8),
            bn1=L.BatchNormalization(self.gf_dim * 4),
            bn2=L.BatchNormalization(self.gf_dim * 2),
            bn3=L.BatchNormalization(self.gf_dim),

            ### for text condition
            lc_mu=L.Linear(None, self.ef_dim, initialW=w),
            lc_var=L.Linear(None, self.ef_dim, initialW=w),
        )

    def make_hidden(self, batchsize):
        return numpy.random.normal(0, 1, (batchsize, self.n_hidden)) \
            .astype(numpy.float32)

    def __call__(self, z, c, test=False):
        ### text augmentation
        hc_mu = F.leaky_relu(self.lc_mu(c))
        hc_var = F.leaky_relu(self.lc_var(c))
        h_c = F.gaussian(hc_mu, hc_var)

        ### concate z and c
        h = F.concat((z, h_c))

        ### generate image
        h = F.reshape(F.relu(self.bn0(self.l0(h), test=test)), (z.data.shape[0], self.gf_dim * 8, self.s16, self.s16))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = F.tanh(self.dc4(h))
        if test:
            return x
        else:
            return x, hc_mu, hc_var

class DiscriminatorSimple(chainer.Chain):
    def __init__(self, img_size=(64,64), df_dim=64, ef_dim=128):
        self.s = img_size[0]
        self.s16 = self.s // 16
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        w = chainer.initializers.Normal(0.02)
        super(DiscriminatorSimple, self).__init__(
            ###for generated image
            c1=L.Convolution2D(None, self.df_dim, 4, 2, 1, initialW=w),
            c2=L.Convolution2D(None, self.df_dim*2, 4, 2, 1, initialW=w),
            c3=L.Convolution2D(None, self.df_dim*4, 4, 2, 1, initialW=w),
            c4=L.Convolution2D(None, self.df_dim*8, 4, 2, 1, initialW=w),
            bn2=L.BatchNormalization(self.df_dim * 2),
            bn3=L.BatchNormalization(self.df_dim * 4),
            bn4=L.BatchNormalization(self.df_dim * 8),

            ###for text
            l_c=L.Linear(None, self.ef_dim, initialW=w),

            ###after concate
            c5=L.Convolution2D(None, self.df_dim*8, 1, 1, 0, initialW=w),
            bn5=L.BatchNormalization(self.df_dim*8),
            l6=L.Linear(None, 1, initialW=w)
        )

    def __call__(self, x, c, test=False):
        ###image discriminator
        h = F.leaky_relu(self.c1(x))
        h = F.leaky_relu(self.bn2(self.c2(h), test=test))
        h = F.leaky_relu(self.bn3(self.c3(h), test=test))
        h = F.leaky_relu(self.bn4(self.c4(h), test=test))
        
        ###text Compression and Spatial Replication
        h_c = F.leaky_relu(self.l_c(c))
        h_c = F.expand_dims(h_c, axis=2)
        h_c = F.expand_dims(h_c, axis=2)
        h_c = F.tile(h_c, (1, 1, self.s16, self.s16))
    
        h = F.concat((h, h_c))

        ###after concatenate
        h = F.leaky_relu(self.bn5(self.c5(h), test=test))
        return self.l6(h)
