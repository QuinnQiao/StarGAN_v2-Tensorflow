# import os
# import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
# gpu_device_name = tf.test.gpu_device_name()
# print(gpu_device_name)

# from utils import *
# from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch # tf 1.13

# with tf.Session() as sess:

#     img_class = Image_data(4, 4, 3, '../../datasets/cat2dog/', ['validA', 'validB'], True)
#     img_class.preprocess()

#     dataset_num = len(img_class.image)
#     print("Dataset number : ", dataset_num)

#     img_and_label = tf.data.Dataset.from_tensor_slices((img_class.image, img_class.label))

#     img_and_label = img_and_label.apply(shuffle_and_repeat(dataset_num)).apply(
#         map_and_batch(img_class.image_processing, 2, num_parallel_batches=4,drop_remainder=True))

#     img_and_label_iterator = img_and_label.make_one_shot_iterator()

#     x_real, label_org = img_and_label_iterator.get_next()

#     print('\n\n')
#     print(x_real.shape)
#     print(label_org.shape)

#     def map(x, y, scope='map'):
#         with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#             x = tf.layers.flatten(x)
#             x = tf.layers.dense(x, 2)

#             bs = x.shape[0]
#             x = tf.reshape(x, (bs, 2, 1))

#             index = tf.reshape(tf.range(bs), (bs, 1))
#             y = tf.reshape(y, (bs, 1))
#             index = tf.concat([index, y], axis=1)

#             return x, tf.gather_nd(x, index)

#     with tf.variable_scope(tf.get_variable_scope(), reuse=False):
#         all_z, part_z = map(x_real, label_org)
#         tx = np.zeros((2, 4, 4, 3), dtype=np.float32)
#         ty = np.zeros((2, 1), dtype=np.int32)
#         tx[0, 0, 0, 0] = 1
#         tx[1, 0, 0, 1] = 1
#         t_all, t_part = map(tf.constant(tx), tf.constant(ty))


#     tf.global_variables_initializer().run()
#     for i in range(5):
#         print('\n{}:\n'.format(i))
#         in_x, in_y, out_a, out_p, t_a, t_p = sess.run([x_real, label_org, all_z, part_z, t_all, t_part])
#         print('y:', in_y)
#         print('out:', out_a)
#         print('out[y]:', out_p)
#         print('w:', t_a)
#         print('w[0]', t_p)




# import tensorflow as tf

# a1 = tf.reshape(tf.range(6), (3,2))
# b1 = tf.gather(a1, 1)

# a2 = tf.reshape(tf.range(18), (3,3,2))
# b2 = tf.gather_nd(a2, tf.constant([[0,0], [1,1], [2,2]]))
# c2 = tf.gather_nd(a2, tf.constant([[0,1]]))


# print('\n')
# with tf.Session() as sess:
#     x, y = sess.run([b1, b2])
#     print(x)
#     print(y)

#     a = tf.truncated_normal(shape=[3, 16])
#     b = tf.layers.dense(a, units=1)

#     tf.global_variables_initializer().run()
#     x = sess.run(b)
#     print(x.shape)


from test import StarGAN_v2
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of StarGAN_v2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test or refer_test ?')
    parser.add_argument('--dataset', type=str, default='celebA-HQ_gender', help='dataset_name')
    parser.add_argument('--refer_img_path', type=str, default='refer_img.jpg', help='reference image path')
    parser.add_argument('--refer_img_label', type=int, help='the label of the reference image')

    parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')

    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch size') # each gpu
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of ckpt_save_freq')
    parser.add_argument('--gpu_num', type=int, default=2, help='The number of gpu')
    parser.add_argument('--visible_gpu', type=str, default='0,1', help='Visible Cuda Devices')

    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_iter', type=int, default=50000, help='decay start iteration')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay value')

    parser.add_argument('--adv_weight', type=float, default=1, help='The weight of Adversarial loss')
    parser.add_argument('--sty_weight', type=float, default=1, help='The weight of Style reconstruction loss') # 0.3 for animal
    parser.add_argument('--ds_weight', type=float, default=1, help='The weight of style diversification loss') # 1 for animal
    parser.add_argument('--cyc_weight', type=float, default=1, help='The weight of Cycle-consistency loss') # 0.1 for animal

    parser.add_argument('--r1_weight', type=float, default=1, help='The weight of R1 regularization')
    parser.add_argument('--gp_weight', type=float, default=10, help='The gradient penalty lambda')

    parser.add_argument('--gan_type', type=str, default='gan', help='gan / lsgan / hinge / wgan-gp / wgan-lp / dragan')
    parser.add_argument('--sn', type=str2bool, default=False, help='using spectral norm')

    parser.add_argument('--ch', type=int, default=32, help='base channel number per layer')
    parser.add_argument('--n_layer_1', type=int, default=3, help='The number of resblock')
    parser.add_argument('--n_layer_2', type=int, default=2, help='The number of resblock')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--style_dim', type=int, default=16, help='length of style code')

    parser.add_argument('--num_style', type=int, default=5, help='number of styles to sample')

    parser.add_argument('--img_height', type=int, default=128, help='The height size of image')
    parser.add_argument('--img_width', type=int, default=128, help='The width size of image ')    
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpu

    # open session
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        gan = StarGAN_v2(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        # show_all_variables()

        gan.train()



if __name__ == '__main__':
    main()