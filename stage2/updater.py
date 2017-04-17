#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable
from chainer.functions.loss.vae import gaussian_kl_divergence


class Stage2_Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen_s1, self.gen_s2, self.dis_s2 = kwargs.pop('models')
        self.KL_COEFF = kwargs.pop('KL_COEFF')
        super(Stage2_Updater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real, batchsize):
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake, mu, ln_var, batchsize):
        GEN_loss = F.sum(F.softplus(-y_fake)) / batchsize
        KL_loss = gaussian_kl_divergence(mu, ln_var) / batchsize
        loss = GEN_loss + self.KL_COEFF * KL_loss
        chainer.report({'loss': loss, 'kl_loss': KL_loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        x_real, c_in = tuple(Variable(x) for x in in_arrays)

        xp = chainer.cuda.get_array_module(x_real.data)

        gen_s1, gen_s2, dis_s2 = self.gen_s1, self.gen_s2, self.dis_s2
        batchsize = x_real.data.shape[0]

        ### discriminate 256x256 real image
        y_real = dis_s2(x_real, c_in, test=False)

        ### generate 64x64 image
        z = Variable(xp.asarray(gen_s1.make_hidden(batchsize)))
        x_lr_fake = gen_s1(z, c_in, test=True)
        x_hr_fake, mu, ln_var = gen_s2(x_lr_fake, c_in, test=False)
        
        y_fake = dis_s2(x_hr_fake, c_in, test=False)
        
        dis_optimizer.update(self.loss_dis, dis_s2, y_fake, y_real, batchsize)
        gen_optimizer.update(self.loss_gen, gen_s2, y_fake, mu, ln_var, batchsize)