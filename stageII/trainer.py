from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar
from PIL import Image, ImageDraw, ImageFont


from misc.config import cfg
from misc.utils import mkdir_p

TINY = 1e-8


# reduce_mean normalize also the dimension of the embeddings
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss


class CondGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        self.log_vars = []

        self.hr_image_shape = self.dataset.image_shape
        ratio = self.dataset.hr_lr_ratio
        self.lr_image_shape = [int(self.hr_image_shape[0] / ratio),
                               int(self.hr_image_shape[1] / ratio),
                               self.hr_image_shape[2]]
        print('hr_image_shape', self.hr_image_shape)
        print('lr_image_shape', self.lr_image_shape)

    def build_placeholder(self):
        '''Helper function for init_opt'''
        self.hr_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.hr_image_shape,
            name='real_hr_images')
        self.hr_wrong_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.hr_image_shape,
            name='wrong_hr_images'
        )
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )

        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )
        #
        self.images = tf.image.resize_bilinear(self.hr_images,
                                               self.lr_image_shape[:2])
        self.wrong_images = tf.image.resize_bilinear(self.hr_wrong_images,
                                                     self.lr_image_shape[:2])

    def sample_encoded_context(self, embeddings):
        '''Helper function for init_opt'''
        # Build conditioning augmentation structure for text embedding
        # under different variable_scope: 'g_net' and 'hr_g_net'
        c_mean_logsigma = self.model.generate_condition(embeddings)
        mean = c_mean_logsigma[0]
        if cfg.TRAIN.COND_AUGMENTATION:
            # epsilon = tf.random_normal(tf.shape(mean))
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(c_mean_logsigma[1])
            c = mean + stddev * epsilon

            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        else:
            c = mean
            kl_loss = 0
        # TODO: play with the coefficient for KL
        return c, cfg.TRAIN.COEFF.KL * kl_loss

    def init_opt(self):
        self.build_placeholder()

        with pt.defaults_scope(phase=pt.Phase.train):
            # ####get output from G network####################################
            with tf.variable_scope("g_net"):
                c, kl_loss = self.sample_encoded_context(self.embeddings)
                z = tf.random_normal([self.batch_size, cfg.Z_DIM])
                self.log_vars.append(("hist_c", c))
                self.log_vars.append(("hist_z", z))
                fake_images = self.model.get_generator(tf.concat(1, [c, z]))

            # ####get discriminator_loss and generator_loss ###################
            discriminator_loss, generator_loss =\
                self.compute_losses(self.images,
                                    self.wrong_images,
                                    fake_images,
                                    self.embeddings,
                                    flag='lr')
            generator_loss += kl_loss
            self.log_vars.append(("g_loss_kl_loss", kl_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))

            # #### For hr_g and hr_d #########################################
            with tf.variable_scope("hr_g_net"):
                hr_c, hr_kl_loss = self.sample_encoded_context(self.embeddings)
                self.log_vars.append(("hist_hr_c", hr_c))
                hr_fake_images = self.model.hr_get_generator(fake_images, hr_c)
            # get losses
            hr_discriminator_loss, hr_generator_loss =\
                self.compute_losses(self.hr_images,
                                    self.hr_wrong_images,
                                    hr_fake_images,
                                    self.embeddings,
                                    flag='hr')
            hr_generator_loss += hr_kl_loss
            self.log_vars.append(("hr_g_loss", hr_generator_loss))
            self.log_vars.append(("hr_d_loss", hr_discriminator_loss))

            # #######define self.g_sum, self.d_sum,....########################
            self.prepare_trainer(discriminator_loss, generator_loss,
                                 hr_discriminator_loss, hr_generator_loss)
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            self.sampler()
            self.visualization(cfg.TRAIN.NUM_COPY)
            print("success")

    def sampler(self):
        with tf.variable_scope("g_net", reuse=True):
            c, _ = self.sample_encoded_context(self.embeddings)
            z = tf.random_normal([self.batch_size, cfg.Z_DIM])
            self.fake_images = self.model.get_generator(tf.concat(1, [c, z]))
        with tf.variable_scope("hr_g_net", reuse=True):
            hr_c, _ = self.sample_encoded_context(self.embeddings)
            self.hr_fake_images =\
                self.model.hr_get_generator(self.fake_images, hr_c)

    def compute_losses(self, images, wrong_images,
                       fake_images, embeddings, flag='lr'):
        if flag == 'lr':
            real_logit =\
                self.model.get_discriminator(images, embeddings)
            wrong_logit =\
                self.model.get_discriminator(wrong_images, embeddings)
            fake_logit =\
                self.model.get_discriminator(fake_images, embeddings)
        else:
            real_logit =\
                self.model.hr_get_discriminator(images, embeddings)
            wrong_logit =\
                self.model.hr_get_discriminator(wrong_images, embeddings)
            fake_logit =\
                self.model.hr_get_discriminator(fake_images, embeddings)

        real_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(real_logit,
                                                    tf.ones_like(real_logit))
        real_d_loss = tf.reduce_mean(real_d_loss)
        wrong_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(wrong_logit,
                                                    tf.zeros_like(wrong_logit))
        wrong_d_loss = tf.reduce_mean(wrong_d_loss)
        fake_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(fake_logit,
                                                    tf.zeros_like(fake_logit))
        fake_d_loss = tf.reduce_mean(fake_d_loss)
        if cfg.TRAIN.B_WRONG:
            discriminator_loss =\
                real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        else:
            discriminator_loss = real_d_loss + fake_d_loss
        if flag == 'lr':
            self.log_vars.append(("d_loss_real", real_d_loss))
            self.log_vars.append(("d_loss_fake", fake_d_loss))
            if cfg.TRAIN.B_WRONG:
                self.log_vars.append(("d_loss_wrong", wrong_d_loss))
        else:
            self.log_vars.append(("hr_d_loss_real", real_d_loss))
            self.log_vars.append(("hr_d_loss_fake", fake_d_loss))
            if cfg.TRAIN.B_WRONG:
                self.log_vars.append(("hr_d_loss_wrong", wrong_d_loss))

        generator_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(fake_logit,
                                                    tf.ones_like(fake_logit))
        generator_loss = tf.reduce_mean(generator_loss)
        if flag == 'lr':
            self.log_vars.append(("g_loss_fake", generator_loss))
        else:
            self.log_vars.append(("hr_g_loss_fake", generator_loss))

        return discriminator_loss, generator_loss

    def define_one_trainer(self, loss, learning_rate, key_word):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()
        tarin_vars = [var for var in all_vars if
                      var.name.startswith(key_word)]

        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        trainer = pt.apply_optimizer(opt, losses=[loss], var_list=tarin_vars)
        return trainer

    def prepare_trainer(self, discriminator_loss, generator_loss,
                        hr_discriminator_loss, hr_generator_loss):
        ft_lr_retio = cfg.TRAIN.FT_LR_RETIO
        self.discriminator_trainer =\
            self.define_one_trainer(discriminator_loss,
                                    self.discriminator_lr * ft_lr_retio,
                                    'd_')
        self.generator_trainer =\
            self.define_one_trainer(generator_loss,
                                    self.generator_lr * ft_lr_retio,
                                    'g_')
        self.hr_discriminator_trainer =\
            self.define_one_trainer(hr_discriminator_loss,
                                    self.discriminator_lr,
                                    'hr_d_')
        self.hr_generator_trainer =\
            self.define_one_trainer(hr_generator_loss,
                                    self.generator_lr,
                                    'hr_g_')

        self.ft_generator_trainer = \
            self.define_one_trainer(hr_generator_loss,
                                    self.generator_lr * cfg.TRAIN.FT_LR_RETIO,
                                    'g_')

        self.log_vars.append(("hr_d_learning_rate", self.discriminator_lr))
        self.log_vars.append(("hr_g_learning_rate", self.generator_lr))

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'd': [], 'hr_g': [], 'hr_d': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.scalar_summary(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.scalar_summary(k, v))
            elif k.startswith('hr_g'):
                all_sum['hr_g'].append(tf.scalar_summary(k, v))
            elif k.startswith('hr_d'):
                all_sum['hr_d'].append(tf.scalar_summary(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.histogram_summary(k, v))

        self.g_sum = tf.merge_summary(all_sum['g'])
        self.d_sum = tf.merge_summary(all_sum['d'])
        self.hr_g_sum = tf.merge_summary(all_sum['hr_g'])
        self.hr_d_sum = tf.merge_summary(all_sum['hr_d'])
        self.hist_sum = tf.merge_summary(all_sum['hist'])

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            row_img = [img]  # real image
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(1, row_img))
        imgs = tf.expand_dims(tf.concat(0, stacked_img), 0)
        current_img_summary = tf.image_summary(filename, imgs)
        return current_img_summary, imgs

    def visualization(self, n):
        fake_sum_train, superimage_train =\
            self.visualize_one_superimage(self.fake_images[:n * n],
                                          self.images[:n * n],
                                          n, "train")
        fake_sum_test, superimage_test =\
            self.visualize_one_superimage(self.fake_images[n * n:2 * n * n],
                                          self.images[n * n:2 * n * n],
                                          n, "test")
        self.superimages = tf.concat(0, [superimage_train, superimage_test])
        self.image_summary = tf.merge_summary([fake_sum_train, fake_sum_test])

        hr_fake_sum_train, hr_superimage_train =\
            self.visualize_one_superimage(self.hr_fake_images[:n * n],
                                          self.hr_images[:n * n, :, :, :],
                                          n, "hr_train")
        hr_fake_sum_test, hr_superimage_test =\
            self.visualize_one_superimage(self.hr_fake_images[n * n:2 * n * n],
                                          self.hr_images[n * n:2 * n * n],
                                          n, "hr_test")
        self.hr_superimages =\
            tf.concat(0, [hr_superimage_train, hr_superimage_test])
        self.hr_image_summary =\
            tf.merge_summary([hr_fake_sum_train, hr_fake_sum_test])

    def preprocess(self, x, n):
        # make sure every row with n column have the same embeddings
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def epoch_sum_images(self, sess, n):
        images_train, _, embeddings_train, captions_train, _ =\
            self.dataset.train.next_batch(n * n, cfg.TRAIN.NUM_EMBEDDING)
        images_train = self.preprocess(images_train, n)
        embeddings_train = self.preprocess(embeddings_train, n)

        images_test, _, embeddings_test, captions_test, _ =\
            self.dataset.test.next_batch(n * n, 1)
        images_test = self.preprocess(images_test, n)
        embeddings_test = self.preprocess(embeddings_test, n)

        images = np.concatenate([images_train, images_test], axis=0)
        embeddings =\
            np.concatenate([embeddings_train, embeddings_test], axis=0)

        if self.batch_size > 2 * n * n:
            images_pad, _, embeddings_pad, _, _ =\
                self.dataset.test.next_batch(self.batch_size - 2 * n * n, 1)
            images = np.concatenate([images, images_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)

        feed_out = [self.superimages, self.image_summary,
                    self.hr_superimages, self.hr_image_summary]
        feed_dict = {self.hr_images: images,
                     self.embeddings: embeddings}
        gen_samples, img_summary, hr_gen_samples, hr_img_summary =\
            sess.run(feed_out, feed_dict)

        # save images generated for train and test captions
        scipy.misc.imsave('%s/lr_fake_train.jpg' %
                          (self.log_dir), gen_samples[0])
        scipy.misc.imsave('%s/lr_fake_test.jpg' %
                          (self.log_dir), gen_samples[1])
        #
        scipy.misc.imsave('%s/hr_fake_train.jpg' %
                          (self.log_dir), hr_gen_samples[0])
        scipy.misc.imsave('%s/hr_fake_test.jpg' %
                          (self.log_dir), hr_gen_samples[1])

        # pfi_train = open(self.log_dir + "/train.txt", "w")
        pfi_test = open(self.log_dir + "/test.txt", "w")
        for row in range(n):
            # pfi_train.write('\n***row %d***\n' % row)
            # pfi_train.write(captions_train[row * n])

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_test[row * n])
        # pfi_train.close()
        pfi_test.close()

        return img_summary, hr_img_summary

    def build_model(self, sess):
        self.init_opt()

        sess.run(tf.initialize_all_variables())
        if len(self.model_path) > 0:
            print("Reading model parameters from %s" % self.model_path)
            all_vars = tf.trainable_variables()
            # all_vars = tf.all_variables()
            restore_vars = []
            for var in all_vars:
                if var.name.startswith('g_') or var.name.startswith('d_'):
                    restore_vars.append(var)
                    # print(var.name)
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, self.model_path)

            istart = self.model_path.rfind('_') + 1
            iend = self.model_path.rfind('.')
            counter = self.model_path[istart:iend]
            counter = int(counter)
        else:
            print("Created model with fresh parameters.")
            counter = 0
        return counter

    def train_one_step(self, generator_lr,
                       discriminator_lr,
                       counter, summary_writer, log_vars, sess):
        # training d
        hr_images, hr_wrong_images, embeddings, _, _ =\
            self.dataset.train.next_batch(self.batch_size,
                                          cfg.TRAIN.NUM_EMBEDDING)
        feed_dict = {self.hr_images: hr_images,
                     self.hr_wrong_images: hr_wrong_images,
                     self.embeddings: embeddings,
                     self.generator_lr: generator_lr,
                     self.discriminator_lr: discriminator_lr
                     }
        if cfg.TRAIN.FINETUNE_LR:
            # train d1
            feed_out_d = [self.hr_discriminator_trainer,
                          self.hr_d_sum,
                          log_vars,
                          self.hist_sum]
            ret_list = sess.run(feed_out_d, feed_dict)
            summary_writer.add_summary(ret_list[1], counter)
            log_vals = ret_list[2]
            summary_writer.add_summary(ret_list[3], counter)
            # train g1 and finetune g0 with the loss of g1
            feed_out_g = [self.hr_generator_trainer,
                          self.ft_generator_trainer,
                          self.hr_g_sum]
            _, _, hr_g_sum = sess.run(feed_out_g, feed_dict)
            summary_writer.add_summary(hr_g_sum, counter)
            # finetune d0 with the loss of d0
            feed_out_d = [self.discriminator_trainer, self.d_sum]
            _, d_sum = sess.run(feed_out_d, feed_dict)
            summary_writer.add_summary(d_sum, counter)
            # finetune g0 with the loss of g0
            feed_out_g = [self.generator_trainer, self.g_sum]
            _, g_sum = sess.run(feed_out_g, feed_dict)
            summary_writer.add_summary(g_sum, counter)
        else:
            # train d1
            feed_out_d = [self.hr_discriminator_trainer,
                          self.hr_d_sum,
                          log_vars,
                          self.hist_sum]
            ret_list = sess.run(feed_out_d, feed_dict)
            summary_writer.add_summary(ret_list[1], counter)
            log_vals = ret_list[2]
            summary_writer.add_summary(ret_list[3], counter)
            # train g1
            feed_out_g = [self.hr_generator_trainer,
                          self.hr_g_sum]
            _, hr_g_sum = sess.run(feed_out_g, feed_dict)
            summary_writer.add_summary(hr_g_sum, counter)

        return log_vals

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                saver = tf.train.Saver(tf.all_variables(),
                                       keep_checkpoint_every_n_hours=5)

                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(self.log_dir,
                                                        sess.graph)

                if cfg.TRAIN.FINETUNE_LR:
                    keys = ["hr_d_loss", "hr_g_loss", "d_loss", "g_loss"]
                else:
                    keys = ["d_loss", "g_loss"]
                log_vars = []
                log_keys = []
                for k, v in self.log_vars:
                    if k in keys:
                        log_vars.append(v)
                        log_keys.append(k)
                generator_lr = cfg.TRAIN.GENERATOR_LR
                discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
                lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
                number_example = self.dataset.train._num_examples
                updates_per_epoch = int(number_example / self.batch_size)
                # int((counter + lr_decay_step/2) / lr_decay_step)
                decay_start = cfg.TRAIN.PRETRAINED_EPOCH
                epoch_start = int(counter / updates_per_epoch)
                for epoch in range(epoch_start, self.max_epoch):
                    widgets = ["epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch,
                                       widgets=widgets)
                    pbar.start()

                    if epoch % lr_decay_step == 0 and epoch > decay_start:
                        generator_lr *= 0.5
                        discriminator_lr *= 0.5

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        pbar.update(i)
                        log_vals = self.train_one_step(generator_lr,
                                                       discriminator_lr,
                                                       counter, summary_writer,
                                                       log_vars, sess)
                        all_log_vals.append(log_vals)
                        # save checkpoint
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_path = "%s/%s_%s.ckpt" %\
                                             (self.checkpoint_dir,
                                              self.exp_name,
                                              str(counter))
                            fn = saver.save(sess, snapshot_path)
                            print("Model saved in file: %s" % fn)

                    img_summary, img_summary2 =\
                        self.epoch_sum_images(sess, cfg.TRAIN.NUM_COPY)
                    summary_writer.add_summary(img_summary, counter)
                    summary_writer.add_summary(img_summary2, counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    dic_logs = {}
                    for k, v in zip(log_keys, avg_log_vals):
                        dic_logs[k] = v
                        # print(k, v)

                    log_line = "; ".join("%s: %s" %
                                         (str(k), str(dic_logs[k]))
                                         for k in dic_logs)
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")

    def drawCaption(self, img, caption):
        img_txt = Image.fromarray(img)
        # get a font
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
        # get a drawing context
        d = ImageDraw.Draw(img_txt)

        # draw text, half opacity
        d.text((10, 256), 'Stage-I', font=fnt, fill=(255, 255, 255, 255))
        d.text((10, 512), 'Stage-II', font=fnt, fill=(255, 255, 255, 255))
        if img.shape[0] > 832:
            d.text((10, 832), 'Stage-I', font=fnt, fill=(255, 255, 255, 255))
            d.text((10, 1088), 'Stage-II', font=fnt, fill=(255, 255, 255, 255))

        idx = caption.find(' ', 60)
        if idx == -1:
            d.text((256, 10), caption, font=fnt, fill=(255, 255, 255, 255))
        else:
            cap1 = caption[:idx]
            cap2 = caption[idx+1:]
            d.text((256, 10), cap1, font=fnt, fill=(255, 255, 255, 255))
            d.text((256, 60), cap2, font=fnt, fill=(255, 255, 255, 255))

        return img_txt

    def save_super_images(self, images, sample_batchs, hr_sample_batchs,
                          savenames, captions_batchs,
                          sentenceID, save_dir, subset):
        # batch_size samples for each embedding
        # Up to 16 samples for each text embedding/sentence
        numSamples = len(sample_batchs)
        for j in range(len(savenames)):
            s_tmp = '%s-1real-%dsamples/%s/%s' %\
                (save_dir, numSamples, subset, savenames[j])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            # First row with up to 8 samples
            real_img = (images[j] + 1.0) * 127.5
            img_shape = real_img.shape
            padding0 = np.zeros(img_shape)
            padding = np.zeros((img_shape[0], 20, 3))

            row1 = [padding0, real_img, padding]
            row2 = [padding0, real_img, padding]
            for i in range(np.minimum(8, numSamples)):
                lr_img = sample_batchs[i][j]
                hr_img = hr_sample_batchs[i][j]
                hr_img = (hr_img + 1.0) * 127.5
                re_sample = scipy.misc.imresize(lr_img, hr_img.shape[:2])
                row1.append(re_sample)
                row2.append(hr_img)
            row1 = np.concatenate(row1, axis=1)
            row2 = np.concatenate(row2, axis=1)
            superimage = np.concatenate([row1, row2], axis=0)

            # Second 8 samples with up to 8 samples
            if len(sample_batchs) > 8:
                row1 = [padding0, real_img, padding]
                row2 = [padding0, real_img, padding]
                for i in range(8, len(sample_batchs)):
                    lr_img = sample_batchs[i][j]
                    hr_img = hr_sample_batchs[i][j]
                    hr_img = (hr_img + 1.0) * 127.5
                    re_sample = scipy.misc.imresize(lr_img, hr_img.shape[:2])
                    row1.append(re_sample)
                    row2.append(hr_img)
                row1 = np.concatenate(row1, axis=1)
                row2 = np.concatenate(row2, axis=1)
                super_row = np.concatenate([row1, row2], axis=0)
                superimage2 = np.zeros_like(superimage)
                superimage2[:super_row.shape[0],
                            :super_row.shape[1],
                            :super_row.shape[2]] = super_row
                mid_padding = np.zeros((64, superimage.shape[1], 3))
                superimage = np.concatenate([superimage, mid_padding,
                                             superimage2], axis=0)

            top_padding = np.zeros((128, superimage.shape[1], 3))
            superimage =\
                np.concatenate([top_padding, superimage], axis=0)

            captions = captions_batchs[j][sentenceID]
            fullpath = '%s_sentence%d.jpg' % (s_tmp, sentenceID)
            superimage = self.drawCaption(np.uint8(superimage), captions)
            scipy.misc.imsave(fullpath, superimage)

    def eval_one_dataset(self, sess, dataset, save_dir, subset='train'):
        count = 0
        print('num_examples:', dataset._num_examples)
        while count < dataset._num_examples:
            start = count % dataset._num_examples
            images, embeddings_batchs, savenames, captions_batchs =\
                dataset.next_batch_test(self.batch_size, start, 1)

            print('count = ', count, 'start = ', start)
            # the i-th sentence/caption
            for i in range(len(embeddings_batchs)):
                samples_batchs = []
                hr_samples_batchs = []
                # Generate up to 16 images for each sentence,
                # with randomness from noise z and conditioning augmentation.
                numSamples = np.minimum(16, cfg.TRAIN.NUM_COPY)
                for j in range(numSamples):
                    hr_samples, samples =\
                        sess.run([self.hr_fake_images, self.fake_images],
                                 {self.embeddings: embeddings_batchs[i]})
                    samples_batchs.append(samples)
                    hr_samples_batchs.append(hr_samples)
                self.save_super_images(images, samples_batchs,
                                       hr_samples_batchs,
                                       savenames, captions_batchs,
                                       i, save_dir, subset)

            count += self.batch_size

    def evaluate(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                if self.model_path.find('.ckpt') != -1:
                    self.init_opt()
                    print("Reading model parameters from %s" % self.model_path)
                    saver = tf.train.Saver(tf.all_variables())
                    saver.restore(sess, self.model_path)
                    # self.eval_one_dataset(sess, self.dataset.train,
                    #                       self.log_dir, subset='train')
                    self.eval_one_dataset(sess, self.dataset.test,
                                          self.log_dir, subset='test')
                else:
                    print("Input a valid model path.")
