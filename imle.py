#! -*- coding: utf-8 -*-
#《Implicit Maximum Likelihood Estimation》例子

import numpy as np
import scipy as sp
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import RMSprop
from keras.callbacks import Callback
import os,json
import warnings
warnings.filterwarnings("ignore") # 忽略keras带来的满屏警告


if not os.path.exists('samples'):
    os.mkdir('samples')


imgs = glob.glob('../../CelebA-HQ/train/*.png')
np.random.shuffle(imgs)
img_dim = 128
z_dim = 128
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 8
f_size = img_dim // 2**(num_layers + 1)
batch_size = 128


def imread(f, mode='gan'):
    x = misc.imread(f, mode='RGB')
    if mode == 'gan':
        x = misc.imresize(x, (img_dim, img_dim))
        x = x.astype(np.float32)
        return x / 255 * 2 - 1
    elif mode == 'fid':
        x = misc.imresize(x, (299, 299))
        return x.astype(np.float32)


class img_generator:
    """图片迭代器，方便重复调用
    """
    def __init__(self, imgs, mode='gan', batch_size=64):
        self.imgs = imgs
        self.batch_size = batch_size
        self.mode = mode
        if len(imgs) % batch_size == 0:
            self.steps = len(imgs) // batch_size
        else:
            self.steps = len(imgs) // batch_size + 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        X = []
        while True:
            np.random.shuffle(self.imgs)
            for i,f in enumerate(self.imgs):
                X.append(imread(f, self.mode))
                if len(X) == self.batch_size or i == len(self.imgs)-1:
                    X = np.array(X)
                    if self.mode == 'gan':
                        Z = np.random.randn(len(X), z_dim)
                        self.XZ = [X, Z]
                        yield [X, Z], None
                    elif self.mode == 'fid':
                        yield X
                    X = []


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(f_size**2 * max_num_channels)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((f_size, f_size, max_num_channels))(z)

for i in range(num_layers):
    num_channels = max_num_channels // 2**(i + 1)
    z = Conv2DTranspose(num_channels,
                        (5, 5),
                        strides=(2, 2),
                        padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

z = Conv2DTranspose(3,
                    (5, 5),
                    strides=(2, 2),
                    padding='same')(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()


# 整合模型
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))

x_real = x_in
x_fake = g_model(z_in)

train_model = Model([x_in, z_in], [x_real, x_fake])

x_real = K.expand_dims(x_real, 1)
x_fake = K.expand_dims(x_fake, 0)
loss = K.sum(x_real**2, [2, 3, 4]) + K.sum(x_fake**2, [2, 3, 4]) - 2 * K.sum(x_real * x_fake, [2, 3, 4])
loss = K.mean(K.min(loss, 1))

train_model.add_loss(loss)
train_model.compile(optimizer=RMSprop(1e-4))

train_model.summary()


# 采样函数
def sample(path, n=9, z_samples=None):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    if z_samples is None:
        z_samples = np.random.randn(n**2, z_dim)
    for i in range(n):
        for j in range(n):
            z_sample = z_samples[[i * n + j]]
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)


class Trainer(Callback):
    def __init__(self):
        self.batch = 0
        self.n_size = 9
        self.iters_per_sample = 100
        self.Z = np.random.randn(self.n_size**2, z_dim)
    def on_batch_end(self, batch, logs=None):
        if self.batch % self.iters_per_sample == 0:
            sample('samples/test_%s.png' % self.batch,
                self.n_size, self.Z)
            train_model.save_weights('./train_model.weights')
        self.batch += 1


if __name__ == '__main__':

    trainer = Trainer()
    img_data = img_generator(imgs, 'gan', batch_size)

    train_model.fit_generator(img_data.__iter__(),
                              steps_per_epoch=len(img_data),
                              epochs=1000,
                              callbacks=[trainer])
