import chainer
import chainer.functions as F
import chainer.links as L

import sys
sys.path.append('../misc')
from custom_opt import CustomConv2D

class Generator(chainer.Chain):
    def __init__(self, img_size=(256,256), n_hidden=100, gf_dim=128, ef_dim=128):
        self.s = img_size[0]
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s // 2), int(self.s // 4), int(self.s // 8), int(self.s // 16)
        w = chainer.initializers.Normal(0.02)
        self.gf_dim = gf_dim
        self.ef_dim = ef_dim
        super(Generator, self).__init__(
            ###encode images
            c1=CustomConv2D(None, gf_dim, 3, 1, 1, bn=False, activation=F.relu),
            c2=CustomConv2D(None, gf_dim * 2, 4, 2, 1, bn=True, activation=F.relu),
            c3=CustomConv2D(None, gf_dim * 4, 4, 2, 1, bn=True, activation=F.relu),
            
            c_joint=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=F.relu),
            
            ###Residual Blocks
            cr1_0=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=F.relu),
            cr1_1=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=None),
            cr2_0=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=F.relu),
            cr2_1=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=None),
            cr3_0=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=F.relu),
            cr3_1=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=None),
            cr4_0=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=F.relu),
            cr4_1=CustomConv2D(None, gf_dim * 4, 3, 1, 1, bn=True, activation=None),

            ###Upsampling
            dc1=CustomConv2D(None, gf_dim * 2, 3, 1, 1, bn=True, activation=F.relu, resize=True),
            dc2=CustomConv2D(None, gf_dim, 3, 1, 1, bn=True, activation=F.relu, resize=True),
            dc3=CustomConv2D(None, gf_dim // 2, 3, 1, 1, bn=True, activation=F.relu, resize=True),
            dc4=CustomConv2D(None, gf_dim // 4, 3, 1, 1, bn=True, activation=F.relu, resize=True),
            c5 =L.Convolution2D(None, 3, 3, 1, 1, initialW=w),
            
            ###text encoder           
            lc_mu=L.Linear(None, self.ef_dim, initialW=w),
            lc_var=L.Linear(None, self.ef_dim, initialW=w)
        )

    def __call__(self, x, c, test=False):        
        ### text encoding
        hc_mu = F.leaky_relu(self.lc_mu(c))
        hc_var = F.leaky_relu(self.lc_var(c))
        h_c = F.gaussian(hc_mu, hc_var)
       
        h_c = F.expand_dims(h_c, axis=2)
        h_c = F.expand_dims(h_c, axis=2)
        h_c = F.tile(h_c, (1, 1, self.s16, self.s16))
        
        ### image encoder
        h = self.c1(x, test=test)
        h = self.c2(h, test=test)
        h = self.c3(h, test=test)

        ### concate text and image
        h = F.concat((h, h_c))
        h = self.c_joint(h, test=test)

        ### residual block
        h0 = self.cr1_0(h, test=test)
        h0 = self.cr1_1(h0, test=test)
        h = F.relu(h + h0)
        h0 = self.cr2_0(h, test=test)
        h0 = self.cr2_1(h0, test=test)
        h = F.relu(h + h0)
        h0 = self.cr3_0(h, test=test)
        h0 = self.cr3_1(h0, test=test)
        h = F.relu(h + h0)
        h0 = self.cr4_0(h, test=test)
        h0 = self.cr4_1(h0, test=test)
        h = F.relu(h + h0)
        
        ### upsampling
        h = self.dc1(h, test=test)
        h = self.dc2(h, test=test)
        h = self.dc3(h, test=test)
        h = self.dc4(h, test=test)
        h = F.tanh(self.c5(h))
        if test:
            return h
        else:
            return h, hc_mu, hc_var

class Discriminator(chainer.Chain):
    def __init__(self, img_size=(64,64), df_dim=64, ef_dim=128):
        self.s = img_size[0]
        self.s16 = int(self.s // 16)
        w = chainer.initializers.Normal(0.02)
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        super(Discriminator, self).__init__(
            ### encoder for 256x256 images
            c1 = CustomConv2D(None, df_dim, 4, 2, 1, bn=False, activation=F.leaky_relu),
            c2 = CustomConv2D(None, df_dim * 2, 4, 2, 1, bn=True, activation=F.leaky_relu),
            c3 = CustomConv2D(None, df_dim * 4, 4, 2, 1, bn=True, activation=F.leaky_relu),
            c4 = CustomConv2D(None, df_dim * 8, 4, 2, 1, bn=True, activation=F.leaky_relu),
            c5 = CustomConv2D(None, df_dim * 16, 4, 2, 1, bn=True, activation=F.leaky_relu),
            c6 = CustomConv2D(None, df_dim * 32, 4, 2, 1, bn=True, activation=F.leaky_relu),
            c7 = CustomConv2D(None, df_dim * 16, 1, 1, 0, bn=True, activation=F.leaky_relu),
            c8 = CustomConv2D(None, df_dim * 8, 1, 1, 0, bn=True, activation=None),

            cr_1 = CustomConv2D(None, df_dim * 2, 1, 1, 0, bn=True, activation=F.leaky_relu),
            cr_2 = CustomConv2D(None, df_dim * 2, 3, 1, 1, bn=True, activation=F.leaky_relu),
            cr_3 = CustomConv2D(None, df_dim * 8, 3, 1, 1, bn=True, activation=None),

            ### encoder for text
            l_c = L.Linear(None, self.ef_dim, initialW=w),

            ### encoder for concated features
            c9 = CustomConv2D(None, df_dim * 8, 1, 1, 0, bn=True, activation=F.leaky_relu),
            l10 = L.Linear(None, 1, initialW=w),
        )

    def __call__(self, x, c, test=False):
        ### encoder for 256x256 images
        h = self.c1(x, test=test)
        h = self.c2(h, test=test)
        h = self.c3(h, test=test)
        h = self.c4(h, test=test)
        h = self.c5(h, test=test)
        h = self.c6(h, test=test)
        h = self.c7(h, test=test)
        h = self.c8(h, test=test)
        
        h0 = self.cr_1(h, test=test)
        h0 = self.cr_2(h, test=test)
        h0 = self.cr_3(h, test=test)
        
        h = F.leaky_relu(h + h0)

        ### encoder for text
        h_c = F.leaky_relu(self.l_c(c))
        h_c = F.expand_dims(h_c, axis=2)
        h_c = F.expand_dims(h_c, axis=2)
        h_c = F.tile(h_c, (1, 1, self.s16, self.s16))
        
        ### concatenate
        h = F.concat((h, h_c))
        
        ### encoder for concatenated features
        h = self.c9(h, test=test)
        h = self.l10(h)
        return h