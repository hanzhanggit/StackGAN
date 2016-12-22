"""
Some codes from
https://github.com/openai/InfoGAN/blob/master/infogan/misc/custom_ops.py
"""
from __future__ import division
from __future__ import print_function

import prettytensor as pt
from tensorflow.python.training import moving_averages
import tensorflow as tf
from prettytensor.pretty_tensor_class import Phase
import numpy as np


class conv_batch_norm(pt.VarStoreMethod):
    """Code modification of:
     http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
     and
     https://github.com/tensorflow/models/blob/master/inception/inception/slim/ops.py"""

    def __call__(self, input_layer, epsilon=1e-5, decay=0.9, name="batch_norm",
                 in_dim=None, phase=Phase.train):
        shape = input_layer.shape
        shp = in_dim or shape[-1]
        with tf.variable_scope(name) as scope:
            self.mean = self.variable('mean', [shp], init=tf.constant_initializer(0.), train=False)
            self.variance = self.variable('variance', [shp], init=tf.constant_initializer(1.0), train=False)

            self.gamma = self.variable("gamma", [shp], init=tf.random_normal_initializer(1., 0.02))
            self.beta = self.variable("beta", [shp], init=tf.constant_initializer(0.))

            if phase == Phase.train:
                mean, variance = tf.nn.moments(input_layer.tensor, [0, 1, 2])
                mean.set_shape((shp,))
                variance.set_shape((shp,))

                update_moving_mean = moving_averages.assign_moving_average(self.mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(self.variance, variance, decay)

                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    normalized_x = tf.nn.batch_norm_with_global_normalization(
                        input_layer.tensor, mean, variance, self.beta, self.gamma, epsilon,
                        scale_after_normalization=True)
            else:
                normalized_x = tf.nn.batch_norm_with_global_normalization(
                    input_layer.tensor, self.mean, self.variance,
                    self.beta, self.gamma, epsilon,
                    scale_after_normalization=True)
            return input_layer.with_tensor(normalized_x, parameters=self.vars)


pt.Register(assign_defaults=('phase'))(conv_batch_norm)


@pt.Register(assign_defaults=('phase'))
class fc_batch_norm(conv_batch_norm):
    def __call__(self, input_layer, *args, **kwargs):
        ori_shape = input_layer.shape
        if ori_shape[0] is None:
            ori_shape[0] = -1
        new_shape = [ori_shape[0], 1, 1, ori_shape[1]]
        x = tf.reshape(input_layer.tensor, new_shape)
        normalized_x = super(self.__class__, self).__call__(input_layer.with_tensor(x), *args, **kwargs)  # input_layer)
        return normalized_x.reshape(ori_shape)


def leaky_rectify(x, leakiness=0.01):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    # import ipdb; ipdb.set_trace()
    return ret


@pt.Register
class custom_conv2d(pt.VarStoreMethod):
    def __call__(self, input_layer, output_dim,
                 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, in_dim=None, padding='SAME',
                 name="conv2d"):
        with tf.variable_scope(name):
            w = self.variable('w', [k_h, k_w, in_dim or input_layer.shape[-1], output_dim],
                              init=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_layer.tensor, w, strides=[1, d_h, d_w, 1], padding=padding)

            # biases = self.variable('biases', [output_dim], init=tf.constant_initializer(0.0))
            # import ipdb; ipdb.set_trace()
            # return input_layer.with_tensor(tf.nn.bias_add(conv, biases), parameters=self.vars)
            return input_layer.with_tensor(conv, parameters=self.vars)


@pt.Register
class custom_deconv2d(pt.VarStoreMethod):
    def __call__(self, input_layer, output_shape,
                 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                 name="deconv2d"):
        output_shape[0] = input_layer.shape[0]
        ts_output_shape = tf.pack(output_shape)
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = self.variable('w', [k_h, k_w, output_shape[-1], input_layer.shape[-1]],
                              init=tf.random_normal_initializer(stddev=stddev))

            try:
                deconv = tf.nn.conv2d_transpose(input_layer, w,
                                                output_shape=ts_output_shape,
                                                strides=[1, d_h, d_w, 1])

            # Support for versions of TensorFlow before 0.7.0
            except AttributeError:
                deconv = tf.nn.deconv2d(input_layer, w, output_shape=ts_output_shape,
                                        strides=[1, d_h, d_w, 1])

            # biases = self.variable('biases', [output_shape[-1]], init=tf.constant_initializer(0.0))
            # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), [-1] + output_shape[1:])
            deconv = tf.reshape(deconv, [-1] + output_shape[1:])

            return deconv


@pt.Register
class custom_fully_connected(pt.VarStoreMethod):
    def __call__(self, input_layer, output_size, scope=None, in_dim=None, stddev=0.02, bias_start=0.0):
        shape = input_layer.shape
        input_ = input_layer.tensor
        try:
            if len(shape) == 4:
                input_ = tf.reshape(input_, tf.pack([tf.shape(input_)[0], np.prod(shape[1:])]))
                input_.set_shape([None, np.prod(shape[1:])])
                shape = input_.get_shape().as_list()

            with tf.variable_scope(scope or "Linear"):
                matrix = self.variable("Matrix", [in_dim or shape[1], output_size], dt=tf.float32,
                                       init=tf.random_normal_initializer(stddev=stddev))
                bias = self.variable("bias", [output_size], init=tf.constant_initializer(bias_start))
                return input_layer.with_tensor(tf.matmul(input_, matrix) + bias, parameters=self.vars)
        except Exception:
            import ipdb; ipdb.set_trace()
