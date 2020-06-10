import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Lambda, GlobalAveragePooling2D, GlobalMaxPooling2D, \
    Multiply, Dense, Reshape
from tensorflow.keras.models import Model
import sys

sys.setrecursionlimit(10000)

'''
最底下的都是原来rcan的函数。凡是函数开头加了''''''的注释的，都是写rcsan添加的
'''

def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters):
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale=2)(x)
    x = Activation('relu')(x)
    return x


def new_ca(input_tensor, filters, reduce=16):
    '''
    Parameters
    ----------
    input_tensor
    filters
    reduce

    Returns
    -------
    input feature 经过maxpool和avgpool（1*1*c），经过sharedMLP（conv-relu-conv）的输出相加经过sigmoid激活输出特征
    '''
    avg_x = GlobalAveragePooling2D()(input_tensor)  #平均池化
    max_x = GlobalMaxPooling2D()(input_tensor)      #最大池化
    avg_x = Reshape((1, 1, filters))(avg_x)         #reshape整理得到1*1*C的tensor
    max_x = Reshape((1, 1, filters))(max_x)
    avg_x = Dense(filters / reduce, activation='relu', kernel_initializer='he_normal', use_bias=False)(avg_x)   #conv-->relu
    max_x = Dense(filters / reduce, activation='relu', kernel_initializer='he_normal', use_bias=False)(max_x)
    return Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(avg_x + max_x)  #conv-->sigmoid, 两个池化tensor相加


def new_sa(input_tensor, kernel_size=7):
    '''

    Parameters
    ----------
    input_tensor
    kernel_size

    Returns
    -------
    input_tensor 经过maxpool和avgpool生成两张h*w*1的tensor，concat成h*w*1的tensor，经过conv7*7卷积后sigmoid激活
    '''
    assert kernel_size in (3, 7)
    padding = 3
    if kernel_size == 3:
        padding = 1
    avg_x = tf.reduce_mean(input_tensor, axis=1, keepdims=True)
    max_x = tf.reduce_max(input_tensor, axis=1, keepdims=True)
    x = tf.concat([avg_x, max_x], concat_dim=1)
    x = Conv2D(2, 1, kernel_size, padding=padding, use_bias=False)(x)
    return tf.nn.sigmoid(x)


def cbam(input_tensor, filters):
    '''

    Parameters
    ----------
    input_tensor
    filters

    Returns
    -------
    input_tensor经过ca，与原输入做一个channel_wise_multiply。再经过sa与输入的特征图（经过ca）做一次乘法输出
    '''
    x = new_ca(input_tensor, filters)
    x = Multiply()([x, input_tensor])
    y = new_sa(x)
    y = Multiply()([y, x])
    return y


def rcsab(input_tensor, filters, scale=0.1):
    '''

    Parameters
    ----------
    input_tensor
    filters
    scale

    Returns
    -------
    代替原来的rcabconv3*3，relu，conv3*3后调用cbam，有些模型中选择残差处理，下面的注释是查到的两种，不用也可以
    '''
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = cbam(x, filters)
    # residual method 1
    # if scale:
    #     x = Lambda(lambda t: t * scale)(x)
    # residual method 2
    # if self.downsample is not None:
    #     residual = self.downsample(x)
    # out += residual
    # out = self.relu(out)
    x = tf.add(x, input_tensor)
    return x


def new_rg(input_tensor, filters, n_rcab=20):
    '''

    Parameters
    ----------
    input_tensor
    filters
    n_rcab

    Returns
    -------
    替换了原来的rcab为rcsab
    '''
    x = input_tensor
    for _ in range(n_rcab):
        x = rcsab(x, filters)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x


def rir(input_tensor, filters, n_rg=10):
    x = input_tensor
    for _ in range(n_rg):
        x = new_rg(x, filters=filters)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x


def generator(filters=64, n_sub_block=2):
    inputs = Input(shape=(None, None, 3))

    x = x_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
    x = rir(x, filters=filters)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x_1, x])

    for _ in range(n_sub_block):
        x = upsample(x, filters)
    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=inputs, outputs=x)

# def ca(input_tensor, filters, reduce=16):
#     x = GlobalAveragePooling2D()(input_tensor)
#     x = Reshape((1, 1, filters))(x)
#     x = Dense(filters / reduce, activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
#     x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
#     x = Multiply()([x, input_tensor])
#     return x
#
#
# def rcab(input_tensor, filters, scale=0.1):
#     x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
#     x = Activation('relu')(x)
#     x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
#     ca(x, filters)
#     if scale:
#         x = Lambda(lambda t: t * scale)(x)
#     x = Add()([x, input_tensor])
#
#     return x
#
#
# def rg(input_tensor, filters, n_rcab=20):
#     x = input_tensor
#     for _ in range(n_rcab):
#         x = rcab(x, filters)
#     x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
#     x = Add()([x, input_tensor])
#
#     return x
