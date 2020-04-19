from ops import *
from utils import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch # tf 1.13

import numpy as np
from glob import glob
from tqdm import tqdm

class StarGAN_v2() :
    def __init__(self, sess, args):
        self.model_name = 'StarGAN_v2'
        self.sess = sess
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = os.path.basename(args.dataset)
        self.dataset_base = args.dataset
        self.augment_flag = args.augment_flag

        self.decay_flag = args.decay_flag
        self.decay_iter = args.decay_iter

        self.gpu_num = args.gpu_num
        self.iteration = args.iteration #// args.gpu_num

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq #// args.gpu_num
        self.save_freq = args.save_freq #// args.gpu_num

        self.init_lr = args.lr
        self.ema_decay = args.ema_decay
        self.ch = args.ch

        self.dataset_path = os.path.join(self.dataset_base, 'train')
        self.label_list = [os.path.basename(x) for x in glob(self.dataset_path + '/*')]
        self.c_dim = len(self.label_list)

        self.refer_img_path = args.refer_img_path
        self.refer_img_label = args.refer_img_label

        """ Weight """
        self.adv_weight = args.adv_weight
        self.sty_weight = args.sty_weight
        self.ds_weight = args.ds_weight
        self.cyc_weight = args.cyc_weight

        self.r1_weight = args.r1_weight
        self.gp_weight = args.gp_weight

        self.sn = args.sn

        """ Generator """
        self.style_dim = args.style_dim
        self.n_layer_1 = args.n_layer_1
        self.n_layer_2 = args.n_layer_2
        self.num_style = args.num_style

        """ Discriminator """
        self.n_critic = args.n_critic

        self.img_height = args.img_height
        self.img_width = args.img_width
        self.img_ch = args.img_ch

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# selected_attrs : ", self.label_list)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# gpu num : ", self.gpu_num)
        print("# iteration : ", self.iteration)
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Generator #####")
        print("# base channel : ", self.ch)
        print("# layer number : ", self.n_layer_1 + self.n_layer_2)

        print()

        print("##### Discriminator #####")
        print("# the number of critic : ", self.n_critic)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, style, scope="generator"):
        channel = self.ch

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) :
            x = conv(x_init, channels=channel, kernel=1, stride=1, use_bias=True, scope='conv_1x1')

            for i in range(self.n_layer_1) :
                x = pre_resblock(x, channels=channel * 2, use_bias=True, scope='down_pre_resblock_' + str(i))
                x = down_sample_avg(x)

                channel = channel * 2

            for i in range(self.n_layer_2) :
                x = pre_resblock(x, channels=channel, use_bias=True, scope='inter_pre_resblock_' + str(i))

            for i in range(self.n_layer_2) :
                gamma1 = fully_connected(style, channel, scope='inter_gamma1_fc_' + str(i))
                beta1 = fully_connected(style, channel, scope='inter_beta1_fc_' + str(i))

                gamma2 = fully_connected(style, channel, scope='inter_gamma2_fc_' + str(i))
                beta2 = fully_connected(style, channel, scope='inter_beta2_fc_' + str(i))

                gamma1 = tf.reshape(gamma1, shape=[gamma1.shape[0], 1, 1, -1])
                beta1 = tf.reshape(beta1, shape=[beta1.shape[0], 1, 1, -1])

                gamma2 = tf.reshape(gamma2, shape=[gamma2.shape[0], 1, 1, -1])
                beta2 = tf.reshape(beta2, shape=[beta2.shape[0], 1, 1, -1])

                x = pre_adaptive_resblock(x, channel, gamma1, beta1, gamma2, beta2, use_bias=True, scope='inter_pre_ada_resblock_' + str(i))

            for i in range(self.n_layer_1) :
                x = up_sample_nearest(x)

                gamma1 = fully_connected(style, channel, scope='up_gamma1_fc_' + str(i))
                beta1 = fully_connected(style, channel, scope='up_beta1_fc_' + str(i))

                gamma2 = fully_connected(style, channel // 2, scope='up_gamma2_fc_' + str(i))
                beta2 = fully_connected(style, channel // 2, scope='up_beta2_fc_' + str(i))

                gamma1 = tf.reshape(gamma1, shape=[gamma1.shape[0], 1, 1, -1])
                beta1 = tf.reshape(beta1, shape=[beta1.shape[0], 1, 1, -1])

                gamma2 = tf.reshape(gamma2, shape=[gamma2.shape[0], 1, 1, -1])
                beta2 = tf.reshape(beta2, shape=[beta2.shape[0], 1, 1, -1])

                x = pre_adaptive_resblock(x, channel // 2, gamma1, beta1, gamma2, beta2, use_bias=True, scope='up_pre_ada_resblock_' + str(i))

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=1, stride=1, use_bias=True, scope='return_image')

            return x

    def style_encoder(self, x_init, label, scope="style_encoder"):
        channel = self.ch // 2
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = conv(x_init, channels=channel, kernel=1, stride=1, use_bias=True, scope='conv_1x1')

            for i in range(self.n_layer_1):
                x = pre_resblock_no_norm_relu(x, channels=channel * 2, use_bias=True, scope='down_pre_resblock_' + str(i))
                x = down_sample_avg(x)

                channel = channel * 2

            channel = channel * 2

            for i in range(self.n_layer_2) :
                x = pre_resblock_no_norm_relu(x, channels=channel, use_bias=True, scope='down_pre_resblock_' + str(i + 4))
                x = down_sample_avg(x)

            kernel_size = int(self.img_height / np.power(2, self.n_layer_1 + self.n_layer_2))

            x = relu(x)
            x = conv(x, channel, kernel=kernel_size, stride=1, use_bias=True, scope='conv_g_kernel')
            x = relu(x)

            bs = x_init.shape[0]
            style = fully_connected(x, units=64 * self.c_dim, use_bias=True, scope='style_fc')
            style = tf.reshape(style, (bs, self.c_dim, 64)) # bs * c_dim * 64

            index = tf.reshape(tf.range(bs), (bs, 1))
            label = tf.reshape(label, (bs, 1))
            index = tf.concat([index, label], axis=1) # bs * 2

            return tf.gather_nd(style, index) # bs * 64

    def mapping_network(self, latent_z, label, scope='mapping_network'):
        channel = self.ch * pow(2, self.n_layer_1)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = latent_z

            # for i in range(self.n_layer_1 + self.n_layer_2):
            for i in range(self.n_layer_1 + self.n_layer_2 + 1):
                x = fully_connected(x, units=channel, use_bias=True, lr_mul=0.01, scope='fc_' + str(i))
                x = relu(x)

            bs = latent_z.shape[0]
            style = fully_connected(x, units=64 * self.c_dim, use_bias=True, lr_mul=0.01, scope='style_fc')
            style = tf.reshape(style, (bs, self.c_dim, 64)) # bs * c_dim * 64

            index = tf.reshape(tf.range(bs), (bs, 1))
            label = tf.reshape(label, (bs, 1))
            index = tf.concat([index, label], axis=1) # bs * 2

            return tf.gather_nd(style, index) # bs * 64

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, label, scope="discriminator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = conv(x_init, channels=channel, kernel=1, stride=1, use_bias=True, scope='conv_1x1')

            for i in range(self.n_layer_1):
                x = pre_resblock_no_norm_lrelu(x, channels=channel * 2, use_bias=True, scope='down_pre_resblock_' + str(i))
                x = down_sample_avg(x)

                channel = channel * 2

            channel = channel * 2

            for i in range(self.n_layer_2):
                x = pre_resblock_no_norm_lrelu(x, channels=channel, use_bias=True, scope='down_pre_resblock_' + str(i + 4))
                x = down_sample_avg(x)

            kernel_size = int(self.img_height / np.power(2, self.n_layer_1 + self.n_layer_2))

            x = lrelu(x, 0.2)
            x = conv(x, channel, kernel=kernel_size, stride=1, use_bias=True, scope='conv_g_kernel')
            x = lrelu(x, 0.2)

            bs = x_init.shape[0]
            logit = fully_connected(x, units=self.c_dim, use_bias=True, scope='dis_logit_fc') # bs * c_dim
            logit = tf.reshape(logit, (bs, self.c_dim, 1))

            index = tf.reshape(tf.range(bs), (bs, 1))
            label = tf.reshape(label, (bs, 1))
            index = tf.concat([index, label], axis=1) # bs * 2

            return tf.gather_nd(logit, index)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):

        self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)

        """ Input Image"""
        img_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, self.label_list,
                               self.augment_flag)
        img_class.preprocess()

        dataset_num = len(img_class.image)
        print("Dataset number : ", dataset_num)

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.ds_weight_placeholder = tf.placeholder(tf.float32, name='ds_weight')


        img_and_label = tf.data.Dataset.from_tensor_slices((img_class.image, img_class.label))

        gpu_device = '/gpu:0'
        img_and_label = img_and_label.apply(shuffle_and_repeat(dataset_num)).apply(
            map_and_batch(img_class.image_processing, self.batch_size * self.gpu_num, num_parallel_batches=16,
                          drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))

        img_and_label_iterator = img_and_label.make_one_shot_iterator()

        self.x_real, label_org = img_and_label_iterator.get_next() # [bs, 256, 256, 3], [bs, 1]
        label_trg = tf.random_uniform(shape=tf.shape(label_org), minval=0, maxval=self.c_dim, dtype=tf.int32) # Target domain labels

        """ split """
        x_real_gpu_split = tf.split(self.x_real, num_or_size_splits=self.gpu_num, axis=0)
        label_org_gpu_split = tf.split(label_org, num_or_size_splits=self.gpu_num, axis=0)
        label_trg_gpu_split = tf.split(label_trg, num_or_size_splits=self.gpu_num, axis=0)
        random_style_code_split = tf.split(tf.random_normal(shape=[self.batch_size * self.gpu_num, self.style_dim]), 
                                            num_or_size_splits=self.gpu_num, axis=0)

        ##############################################################################################################################
        d_adv_loss_per_gpu1 = []
        d_gp_loss_per_gpu1 = []

        for gpu_id in range(self.gpu_num):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):

                    x_real_each = x_real_gpu_split[gpu_id] # [bs. h, w, 3]
                    label_org_each = label_org_gpu_split[gpu_id] # [bs, 1]
                    label_trg_each = label_trg_gpu_split[gpu_id]
                    random_style_code = random_style_code_split[gpu_num]

                    ''' Define Generator, Discriminator '''

                    random_style = self.mapping_network(random_style_code, label_trg_each)

                    x_fake = self.generator(x_real_each, random_style) # for adversarial objective

                    real_logit = self.discriminator(x_real_each, label_org_each)
                    fake_logit = self.discriminator(x_fake, label_trg_each)

                    ''' Define loss '''

                    d_adv_loss = self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit)
                    d_simple_gp = self.adv_weight * simple_gp(real_logit, fake_logit, x_real_each, x_fake, r1_gamma=self.r1_weight, r2_gamma=0.0)
                    d_simple_gp = tf.reduce_sum(d_simple_gp, axis=[1, 2, 3])

                    d_adv_loss_per_gpu1.append(d_adv_loss)
                    d_gp_loss_per_gpu1.append(d_simple_gp)

        self.d_adv_loss1 = tf.concat(d_adv_loss_per_gpu1, axis=0)
        self.d_gp_loss1 = tf.concat(d_gp_loss_per_gpu1, axis=0)


        ##############################################################################################################################
        d_adv_loss_per_gpu2 = []
        d_gp_loss_per_gpu2 = []

        for gpu_id in range(self.gpu_num):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):

                    x_real_split = tf.split(x_real_gpu_split[gpu_id], num_or_size_splits=self.batch_size, axis=0)
                    label_org_split = tf.split(label_org_gpu_split[gpu_id], num_or_size_splits=self.batch_size, axis=0)
                    label_trg_split = tf.split(label_trg_gpu_split[gpu_id], num_or_size_splits=self.batch_size, axis=0)
                    random_style_code_split_gpu = tf.split(random_style_code_split[gpu_id], num_or_size_splits=self.batch_size, axis=0)

                    d_adv_loss = None
                    d_simple_gp = None

                    for each_bs in range(self.batch_size) :
                        """ Define Generator, Discriminator """
                        x_real_each = x_real_split[each_bs] # [1, 256, 256, 3]
                        label_org_each = label_org_split[each_bs]
                        label_trg_each = label_trg_split[each_bs]
                        random_style_code = random_style_code_split_gpu[each_bs]

                        random_style = self.mapping_network(random_style_code, label_trg_each)

                        x_fake = self.generator(x_real_each, random_style) # for adversarial objective

                        real_logit = self.discriminator(x_real_each, label_org_each)
                        fake_logit = self.discriminator(x_fake, label_trg_each)

                        """ Define loss """

                        if each_bs == 0 :

                            d_adv_loss = self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit)
                            d_simple_gp = self.adv_weight * simple_gp(real_logit, fake_logit, x_real_each, x_fake, r1_gamma=self.r1_weight, r2_gamma=0.0)

                        else :

                            d_adv_loss = tf.concat([d_adv_loss, self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit)], axis=0)
                            d_simple_gp = tf.concat([d_simple_gp, self.adv_weight * simple_gp(real_logit, fake_logit, x_real_each, x_fake, r1_gamma=self.r1_weight, r2_gamma=0.0)], axis=0)

                    d_simple_gp = tf.reduce_sum(d_simple_gp, axis=[1, 2, 3])

                    d_adv_loss_per_gpu2.append(d_adv_loss)
                    d_gp_loss_per_gpu2.append(d_simple_gp)

        self.d_adv_loss2 = tf.concat(d_adv_loss_per_gpu2, axis=0)
        self.d_gp_loss2 = tf.concat(d_gp_loss_per_gpu2, axis=0)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=10)

        # summary writer
        # self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_batch_id = checkpoint_counter
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        # loop for epoch
        lr = self.init_lr
        ds_w = self.ds_weight

        train_feed_dict = {
            self.lr : lr,
            self.ds_weight_placeholder: ds_w
        }

        # Update D
        adv1, gp1, adv2, gp2 = self.sess.run([self.d_adv_loss1, self.d_gp_loss1, self.d_adv_loss2, self.d_gp_loss2], 
                                                feed_dict = train_feed_dict)
        print('\n')
        print(adv1)
        print(adv2)
        print(gp1)
        print(gp2)



    @property
    def model_dir(self):

        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}_{}adv_{}sty_{}ds_{}cyc{}".format(self.model_name, self.dataset_name, self.gan_type,
                                                          self.adv_weight, self.sty_weight, self.ds_weight, self.cyc_weight,
                                                          sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0



