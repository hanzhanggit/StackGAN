from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import misc.custom_ops
from misc.custom_ops import leaky_rectify
from misc.config import cfg

# TODO:  Does template.constrct() really shared the computation
# when multipel times of construct are done


class CondGAN(object):
    def __init__(self, lr_imsize, hr_lr_ratio):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.hr_lr_ratio = hr_lr_ratio
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

        self.s = lr_imsize
        print('lr_imsize: ', lr_imsize)
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        if cfg.GAN.NETWORK_TYPE == "default":
            with tf.variable_scope("d_net"):
                self.d_context_template = self.context_embedding()
                self.d_image_template = self.d_encode_image()
                self.d_discriminator_template = self.discriminator()

            with tf.variable_scope("hr_d_net"):
                self.hr_d_context_template = self.context_embedding()
                self.hr_d_image_template = self.hr_d_encode_image()
                self.hr_discriminator_template = self.discriminator()
        else:
            raise NotImplementedError

    # conditioning augmentation structure for text embedding
    # are shared by g and hr_g
    # g and hr_g build this structure separately and do not share parameters
    def generate_condition(self, c_var):
        conditions =\
            (pt.wrap(c_var).
             flatten().
             custom_fully_connected(self.ef_dim * 2).
             apply(leaky_rectify, leakiness=0.2))
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        return [mean, log_sigma]

    # stage I generator (g)
    def generator(self, z_var):
        node1_0 =\
            (pt.wrap(z_var).
             flatten().
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             fc_batch_norm().
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]))
        node1_1 = \
            (node1_0.
             custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(tf.nn.relu))

        node2_0 = \
            (node1.
             # custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2_1 = \
            (node2_0.
             custom_conv2d(self.gf_dim * 1, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 1, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2 = \
            (node2_0.
             apply(tf.add, node2_1).
             apply(tf.nn.relu))

        output_tensor = \
            (node2.
             # custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor

    def get_generator(self, z_var):
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator(z_var)
        else:
            raise NotImplementedError

    # stage II generator (hr_g)
    def residual_block(self, x_c_code):
        node0_0 = pt.wrap(x_c_code)  # -->s4 * s4 * gf_dim * 4
        node0_1 = \
            (pt.wrap(x_c_code).  # -->s4 * s4 * gf_dim * 4
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        output_tensor = \
            (node0_0.
             apply(tf.add, node0_1).
             apply(tf.nn.relu))
        return output_tensor

    def hr_g_encode_image(self, x_var):
        output_tensor = \
            (pt.wrap(x_var).  # -->s * s * 3
             custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).  # s * s * gf_dim
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=4, k_w=4).  # s2 * s2 * gf_dim * 2
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=4, k_w=4).  # s4 * s4 * gf_dim * 4
             conv_batch_norm().
             apply(tf.nn.relu))
        return output_tensor

    def hr_g_joint_img_text(self, x_c_code):
        output_tensor = \
            (pt.wrap(x_c_code).  # -->s4 * s4 * (ef_dim+gf_dim*4)
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).  # s4 * s4 * gf_dim * 4
             conv_batch_norm().
             apply(tf.nn.relu))
        return output_tensor

    def hr_generator(self, x_c_code):
        output_tensor = \
            (pt.wrap(x_c_code).  # -->s4 * s4 * gf_dim*4
             # custom_deconv2d([0, self.s2, self.s2, self.gf_dim * 2], k_h=4, k_w=4).  # -->s2 * s2 * gf_dim*2
             apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s, self.s, self.gf_dim], k_h=4, k_w=4).  # -->s * s * gf_dim
             apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s * 2, self.s * 2, self.gf_dim // 2], k_h=4, k_w=4).  # -->2s * 2s * gf_dim/2
             apply(tf.image.resize_nearest_neighbor, [self.s * 2, self.s * 2]).
             custom_conv2d(self.gf_dim // 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s * 4, self.s * 4, self.gf_dim // 4], k_h=4, k_w=4).  # -->4s * 4s * gf_dim//4
             apply(tf.image.resize_nearest_neighbor, [self.s * 4, self.s * 4]).
             custom_conv2d(self.gf_dim // 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).  # -->4s * 4s * 3
             apply(tf.nn.tanh))
        return output_tensor

    def hr_get_generator(self, x_var, c_code):
        if cfg.GAN.NETWORK_TYPE == "default":
            # image x_var: self.s * self.s *3
            x_code = self.hr_g_encode_image(x_var)  # -->s4 * s4 * gf_dim * 4

            # text c_code: ef_dim
            c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
            c_code = tf.tile(c_code, [1, self.s4, self.s4, 1])

            # combine both --> s4 * s4 * (ef_dim+gf_dim*4)
            x_c_code = tf.concat(3, [x_code, c_code])

            # Joint learning from text and image -->s4 * s4 * gf_dim * 4
            node0 = self.hr_g_joint_img_text(x_c_code)
            node1 = self.residual_block(node0)
            node2 = self.residual_block(node1)
            node3 = self.residual_block(node2)
            node4 = self.residual_block(node3)

            # Up-sampling
            return self.hr_generator(node4)  # -->4s * 4s * 3
        else:
            raise NotImplementedError

    # structure shared by d and hr_d
    # d and hr_d build this structure separately and do not share parameters
    def context_embedding(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template

    def discriminator(self):
        template = \
            (pt.template("input").  # s16 * s16 * 128*9
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # s16 * s16 * 128*8
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))

        return template

    # d-net
    def d_encode_image(self):
        node1_0 = \
            (pt.template("input").  # s * s * 3
             custom_conv2d(self.df_dim, k_h=4, k_w=4).  # s2 * s2 * df_dim
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).  # s4 * s4 * df_dim*2
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).  # s8 * s8 * df_dim*4
             conv_batch_norm().
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).  # s16 * s16 * df_dim*8
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1

    def get_discriminator(self, x_var, c_var):
        x_code = self.d_image_template.construct(input=x_var)  # s16 * s16 * df_dim*8

        c_code = self.d_context_template.construct(input=c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])  # s16 * s16 * ef_dim

        x_c_code = tf.concat(3, [x_code, c_code])
        return self.d_discriminator_template.construct(input=x_c_code)

    # hr_d_net
    def hr_d_encode_image(self):
        node1_0 = \
            (pt.template("input").  # 4s * 4s * 3
             custom_conv2d(self.df_dim, k_h=4, k_w=4).  # 2s * 2s * df_dim
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).  # s * s * df_dim*2
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).  # s2 * s2 * df_dim*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).  # s4 * s4 * df_dim*8
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 16, k_h=4, k_w=4).  # s8 * s8 * df_dim*16
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 32, k_h=4, k_w=4).  # s16 * s16 * df_dim*32
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 16, k_h=1, k_w=1, d_h=1, d_w=1).  # s16 * s16 * df_dim*16
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # s16 * s16 * df_dim*8
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1

    def hr_get_discriminator(self, x_var, c_var):
        x_code = self.hr_d_image_template.construct(input=x_var)  # s16 * s16 * df_dim*8

        c_code = self.hr_d_context_template.construct(input=c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])  # s16 * s16 * ef_dim

        x_c_code = tf.concat(3, [x_code, c_code])
        return self.hr_discriminator_template.construct(input=x_c_code)
