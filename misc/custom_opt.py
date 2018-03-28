#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import utils
from chainer.utils import type_check
from chainer import cuda
from chainer.utils import conv

class CustomConv2D(chainer.Chain):
    def __init__(self, ch0, ch1, ksize, s, p, bn=True, activation=F.relu, resize=False):
        w = chainer.initializers.Normal(0.02)
        self.bn=bn
        self.activation = activation
        self.resize=resize
        layers = {}
        layers['c'] = L.Convolution2D(ch0, ch1, ksize, stride=s, pad=p, initialW=w)
        layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CustomConv2D, self).__init__(**layers)

    def __call__(self, x, test):
        if self.resize:
            ### resize features with NN algorithm            
            h = nn_upsampling(x)
            h = self.c(h)
        else:
            h = self.c(x)
        if self.bn:
            h = self.batchnorm(h, test=test)
        if not self.activation is None:
            h = self.activation(h)
        return h

class NN_Upsampling(chainer.function.Function):
    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, inputs):
        # do forward computation on CPU
        return inputs[0].repeat(2, axis=2).repeat(2, axis=3), 
    
    ### backwardはchainerのaverage pooling 2dのforwardをほぼそのまま活用．
    # 変更点は
    # - col.mean() -> col.sum()
    # - coeff = 1./(self.kh*self.kw) -> coeff = 1. 
    # くらい。あとはksizeやsやpといったパラメーターを2,2,0で決め打ちにしておいた
    def backward_cpu(self, inputs, grad_outputs):
        x = grad_outputs 
        col = conv.im2col_cpu(x[0], 2, 2, 2, 2, 0, 0)
        y = col.sum(axis=(2, 3))
        return y,
    
    def backward_gpu(self, inputs, grad_outputs):
        x = grad_outputs
        n, c, h, w = x[0].shape
        y_h = conv.get_conv_outsize(h, 2, 2, 0)
        y_w = conv.get_conv_outsize(w, 2, 2, 0)
        y = cuda.cupy.empty((n, c, y_h, y_w), dtype=x[0].dtype)
        coeff = 1.
        kern = cuda.elementwise(
            'raw T in, int32 h, int32 w,'
            'int32 out_h, int32 out_w, int32 kh, int32 kw,'
            'int32 sy, int32 sx, int32 ph, int32 pw, T coeff',
            'T out', '''
            int c0    = i / (out_h * out_w);
            int out_y = i / out_w % out_h;
            int out_x = i % out_w;
            int in_y_0 = max(0, out_y * sy - ph);
            int in_y_1 = min(h, out_y * sy + kh - ph);
            int in_x_0 = max(0, out_x * sx - pw);
            int in_x_1 = min(w, out_x * sx + kw - pw);

            T val = 0;
            for (int y = in_y_0; y < in_y_1; ++y) {
              int offset_y = w * (y + h * c0);
              for (int x = in_x_0; x < in_x_1; ++x) {
                val = val + in[x + offset_y];
              }
            }
            out = val * coeff;
            ''', 'avg_pool_fwd')
        kern(x[0].reduced_view(), h, w, y_h, y_w, 2, 2, 
             2, 2, 0, 0, coeff, y)
        return y,

def nn_upsampling(x, use_cudnn=True):
    return NN_Upsampling(use_cudnn)(x)

