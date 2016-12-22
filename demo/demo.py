from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import argparse
import torchfile
from PIL import Image, ImageDraw, ImageFont
import re

from misc.config import cfg, cfg_from_file
from misc.utils import mkdir_p
from stageII.model import CondGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    parser.add_argument('--caption_path', type=str, default=None,
                        help='Path to the file with text sentences')
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args


def sample_encoded_context(embeddings, model, bAugmentation=True):
    '''Helper function for init_opt'''
    # Build conditioning augmentation structure for text embedding
    # under different variable_scope: 'g_net' and 'hr_g_net'
    c_mean_logsigma = model.generate_condition(embeddings)
    mean = c_mean_logsigma[0]
    if bAugmentation:
        # epsilon = tf.random_normal(tf.shape(mean))
        epsilon = tf.truncated_normal(tf.shape(mean))
        stddev = tf.exp(c_mean_logsigma[1])
        c = mean + stddev * epsilon
    else:
        c = mean
    return c


def build_model(sess, embedding_dim, batch_size):
    model = CondGAN(
        lr_imsize=cfg.TEST.LR_IMSIZE,
        hr_lr_ratio=int(cfg.TEST.HR_IMSIZE/cfg.TEST.LR_IMSIZE))

    embeddings = tf.placeholder(
        tf.float32, [batch_size, embedding_dim],
        name='conditional_embeddings')
    with pt.defaults_scope(phase=pt.Phase.test):
        with tf.variable_scope("g_net"):
            c = sample_encoded_context(embeddings, model)
            z = tf.random_normal([batch_size, cfg.Z_DIM])
            fake_images = model.get_generator(tf.concat(1, [c, z]))
        with tf.variable_scope("hr_g_net"):
            hr_c = sample_encoded_context(embeddings, model)
            hr_fake_images = model.hr_get_generator(fake_images, hr_c)

    ckt_path = cfg.TEST.PRETRAINED_MODEL
    if ckt_path.find('.ckpt') != -1:
        print("Reading model parameters from %s" % ckt_path)
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, ckt_path)
    else:
        print("Input a valid model path.")
    return embeddings, fake_images, hr_fake_images


def drawCaption(img, caption):
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


def save_super_images(sample_batchs, hr_sample_batchs,
                      captions_batch, batch_size,
                      startID, save_dir):
    if not os.path.isdir(save_dir):
        print('Make a new folder: ', save_dir)
        mkdir_p(save_dir)

    # Save up to 16 samples for each text embedding/sentence
    img_shape = hr_sample_batchs[0][0].shape
    for j in range(batch_size):
        if not re.search('[a-zA-Z]+', captions_batch[j]):
            continue

        padding = np.zeros(img_shape)
        row1 = [padding]
        row2 = [padding]
        # First row with up to 8 samples
        for i in range(np.minimum(8, len(sample_batchs))):
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
            row1 = [padding]
            row2 = [padding]
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
            superimage =\
                np.concatenate([superimage, mid_padding, superimage2], axis=0)

        top_padding = np.zeros((128, superimage.shape[1], 3))
        superimage =\
            np.concatenate([top_padding, superimage], axis=0)

        fullpath = '%s/sentence%d.jpg' % (save_dir, startID + j)
        superimage = drawCaption(np.uint8(superimage), captions_batch[j])
        scipy.misc.imsave(fullpath, superimage)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.caption_path is not None:
        cfg.TEST.CAPTION_PATH = args.caption_path

    # Load text embeddings generated from the encoder
    cap_path = cfg.TEST.CAPTION_PATH
    t_file = torchfile.load(cap_path)
    captions_list = t_file.raw_txt
    embeddings = np.concatenate(t_file.fea_txt, axis=0)
    num_embeddings = len(captions_list)
    print('Successfully load sentences from: ', cap_path)
    print('Total number of sentences:', num_embeddings)
    print('num_embeddings:', num_embeddings, embeddings.shape)
    # path to save generated samples
    save_dir = cap_path[:cap_path.find('.t7')]
    if num_embeddings > 0:
        batch_size = np.minimum(num_embeddings, cfg.TEST.BATCH_SIZE)

        # Build StackGAN and load the model
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                embeddings_holder, fake_images_opt, hr_fake_images_opt =\
                    build_model(sess, embeddings.shape[-1], batch_size)

                count = 0
                while count < num_embeddings:
                    iend = count + batch_size
                    if iend > num_embeddings:
                        iend = num_embeddings
                        count = num_embeddings - batch_size
                    embeddings_batch = embeddings[count:iend]
                    captions_batch = captions_list[count:iend]

                    samples_batchs = []
                    hr_samples_batchs = []
                    # Generate up to 16 images for each sentence with
                    # randomness from noise z and conditioning augmentation.
                    for i in range(np.minimum(16, cfg.TEST.NUM_COPY)):
                        hr_samples, samples =\
                            sess.run([hr_fake_images_opt, fake_images_opt],
                                     {embeddings_holder: embeddings_batch})
                        samples_batchs.append(samples)
                        hr_samples_batchs.append(hr_samples)
                    save_super_images(samples_batchs,
                                      hr_samples_batchs,
                                      captions_batch,
                                      batch_size,
                                      count, save_dir)
                    count += batch_size

        print('Finish generating samples for %d sentences:' % num_embeddings)
        print('Example sentences:')
        for i in xrange(np.minimum(10, num_embeddings)):
            print('Sentence %d: %s' % (i, captions_list[i]))
