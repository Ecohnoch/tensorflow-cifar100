import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import argparse
import random
import math
import cv2
import os

from model.resnet34 import resnet18, resnet34
from model.resnet50 import resnet50, resnet110, resnet152
from model.serenset50 import se_resnet50, se_resnet110, se_resnet152
from model.densenet import densenet121, densenet161, densenet169, densenet201

from model.seresnet_fixed import get_resnet

from utils import compute_mean_var, norm_images, unpickle, generate_tfrecord, norm_images_using_mean_var


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)

    img = tf.decode_raw(features['image_raw'], tf.float32)
    img = tf.reshape(img, shape=(32, 32, 3))

    img = tf.pad(img, [[4, 4], [4, 4], [0, 0]])
    img = tf.random_crop(img, [32, 32, 3])
    # img = tf.image.random_flip_left_right(img)

    flip = random.getrandbits(1)
    if flip:
        img = img[:, ::-1, :]
    # rot = random.randint(-15, 15)
    # img = tf.contrib.image.rotate(img, rot)
    # img = tf.image.rot90(img, rot)

    label = tf.cast(features['label'], tf.int64)
    return img, label

def parse_test(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    img = tf.decode_raw(features['image_raw'], tf.float32)
    img = tf.reshape(img, shape=(32, 32, 3))

    label = tf.cast(features['label'], tf.int64)
    return img, label

def lr_schedule(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 120:
        return 0.02
    if epoch < 160:
        return 0.004
    if epoch < 200:
        return 0.0008
    

def train(args):
    batch_size = args.batch_size
    epoch = args.epoch
    network = args.network
    opt = args.opt
    train = unpickle(args.train_path)
    test = unpickle(args.test_path)
    train_data = train[b'data']
    test_data  = test[b'data']

    x_train = train_data.reshape(train_data.shape[0], 3, 32, 32)
    x_train = x_train.transpose(0, 2, 3, 1)
    y_train = train[b'fine_labels']


    x_test = test_data.reshape(test_data.shape[0], 3, 32, 32)
    x_test = x_test.transpose(0, 2, 3, 1)
    y_test= test[b'fine_labels']

    x_train = norm_images(x_train)
    x_test = norm_images(x_test)

    print('-------------------------------')
    print('--train/test len: ', len(train_data), len(test_data))
    print('--x_train norm: ', compute_mean_var(x_train))
    print('--x_test norm: ', compute_mean_var(x_test))
    print('--batch_size: ', batch_size)
    print('--epoch: ', epoch)
    print('--network: ', network)
    print('--opt: ', opt)
    print('-------------------------------')

    if not os.path.exists('./trans/tran.tfrecords'):
        generate_tfrecord(x_train, y_train, './trans/', 'tran.tfrecords')
        generate_tfrecord(x_test, y_test, './trans/', 'test.tfrecords')

    dataset = tf.data.TFRecordDataset('./trans/tran.tfrecords')
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(batch_size)
    iterator= dataset.make_initializable_iterator()
    next_element = iterator.get_next() 

    x_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_input = tf.placeholder(tf.int64, [None, ])
    lr = tf.placeholder(tf.float32, [])

    if network == 'resnet50':
        prob = resnet50(x_input, is_training=True, kernel_initializer=tf.orthogonal_initializer())
    elif network == 'resnet34':
        prob = resnet34(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet18':
        prob = resnet18(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet50':
        prob = se_resnet50(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet110':
        prob = resnet110(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet110':
        prob = se_resnet110(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet152':
        prob = se_resnet152(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet152':
        prob = resnet152(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet_fixed':
        prob = get_resnet(x_input, 152, trainable=True, w_init=tf.contrib.layers.xavier_initializer(uniform=False))
    elif network == 'densenet121':
        prob = densenet121(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'densenet169':
        prob = densenet169(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'densenet201':
        prob = densenet201(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'densenet161':
        prob = densenet161(x_input, is_training=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob, labels=y_input))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    # loss = l2_loss * 5e-4 + loss

    if opt == 'adam':
        opt = tf.train.AdamOptimizer(lr)
    elif opt == 'momentum':
        opt = tf.train.MomentumOptimizer(lr, 0.9)
    elif opt == 'nesterov':
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss)

    logit_softmax = tf.nn.softmax(prob)
    acc  = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit_softmax, 1), y_input), tf.float32))

    #-------------------------------Test-----------------------------------------
    if not os.path.exists('./trans/tran.tfrecords'):
        generate_tfrecord(x_test, y_test, './trans/', 'test.tfrecords')
    dataset_test = tf.data.TFRecordDataset('./trans/test.tfrecords')
    dataset_test = dataset_test.map(parse_test)
    dataset_test = dataset_test.shuffle(buffer_size=10000)
    dataset_test = dataset_test.batch(128)
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next() 
    if network == 'resnet50':
        prob_test = resnet50(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet18':
        prob_test = resnet18(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet34':
        prob_test = resnet34(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet50':
        prob_test = se_resnet50(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet110':
        prob_test = resnet110(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet110':
        prob_test = se_resnet110(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet152':
        prob_test = se_resnet152(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet152':
        prob_test = resnet152(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet_fixed':
        prob_test = get_resnet(x_input, 152, type='se_ir', trainable=False, reuse=True)
    elif network == 'densenet121':
        prob_test = densenet121(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'densenet169':
        prob_test = densenet169(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'densenet201':
        prob_test = densenet201(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'densenet161':
        prob_test = densenet161(x_input, is_training=False, reuse=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    logit_softmax_test = tf.nn.softmax(prob_test)
    acc_test = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logit_softmax_test, 1), y_input), tf.float32))
    #----------------------------------------------------------------------------
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    now_lr = 0.001    # Warm Up
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        counter = 0
        for i in range(epoch):
            sess.run(iterator.initializer)
            while True:
                try:
                    batch_train, label_train = sess.run(next_element)
                    _, loss_val, acc_val, lr_val= sess.run([train_op, loss, acc, lr], feed_dict={x_input: batch_train, y_input: label_train, lr: now_lr})

                    counter += 1

                    if counter % 100 == 0:
                        print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val)
                    if counter % 1000 == 0:
                        print('start test ')
                        sess.run(iterator_test.initializer)
                        avg_acc = []
                        while True:
                            try:
                                batch_test, label_test = sess.run(next_element_test)
                                acc_test_val = sess.run(acc_test, feed_dict={x_input: batch_test, y_input: label_test})
                                avg_acc.append(acc_test_val)
                            except tf.errors.OutOfRangeError:
                                print('end test ', np.sum(avg_acc)/len(y_test))
                                if np.sum(avg_acc)/len(y_test) > 0.7:
                                    print("******** 0.7 Got!")
                                    saver = tf.train.Saver(var_list=tf.global_variables())
                                    filename = 'params/distinct/Speaker_vox_iter_{:d}'.format(counter) + '.ckpt'
                                    saver.save(sess, filename)
                                break
                except tf.errors.OutOfRangeError:
                    print('end epoch %d/%d , lr: %f'%(i, epoch, lr_val))
                    now_lr = lr_schedule(i)
                    break

def test(args):
    # train = unpickle('/data/ChuyuanXiong/up/cifar-100-python/train')
    # train_data = train[b'data']
    # x_train = train_data.reshape(train_data.shape[0], 3, 32, 32)
    # x_train = x_train.transpose(0, 2, 3, 1)

    test = unpickle(args.test_path)
    test_data  = test[b'data']

    x_test = test_data.reshape(test_data.shape[0], 3, 32, 32)
    x_test = x_test.transpose(0, 2, 3, 1)
    y_test= test[b'fine_labels']

    x_test = norm_images(x_test)
    # x_test = norm_images_using_mean_var(x_test, *compute_mean_var(x_train))

    network = args.network
    ckpt = args.ckpt

    x_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_input = tf.placeholder(tf.int64, [None, ])
    #-------------------------------Test-----------------------------------------
    if not os.path.exists('./trans/test.tfrecords'):
        generate_tfrecord(x_test, y_test, './trans/', 'test.tfrecords')
    dataset_test = tf.data.TFRecordDataset('./trans/test.tfrecords')
    dataset_test = dataset_test.map(parse_test)
    dataset_test = dataset_test.shuffle(buffer_size=10000)
    dataset_test = dataset_test.batch(128)
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next() 
    if network == 'resnet50':
        prob_test = resnet50(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet18':
        prob_test = resnet18(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet34':
        prob_test = resnet34(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet50':
        prob_test = se_resnet50(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet110':
        prob_test = resnet110(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet110':
        prob_test = se_resnet110(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet152':
        prob_test = se_resnet152(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'resnet152':
        prob_test = resnet152(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'seresnet_fixed':
        prob_test = get_resnet(x_input, 152, type='se_ir', trainable=False, reuse=True)
    elif network == 'densenet121':
        prob_test = densenet121(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'densenet169':
        prob_test = densenet169(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'densenet201':
        prob_test = densenet201(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    elif network == 'densenet161':
        prob_test = densenet161(x_input, is_training=False, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    
    # prob_test = tf.layers.dense(prob_test, 100, reuse=True, name='before_softmax')
    logit_softmax_test = tf.nn.softmax(prob_test)
    acc_test = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logit_softmax_test, 1), y_input), tf.float32))

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    saver = tf.train.Saver(var_list=var_list)
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True



    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt)
        sess.run(iterator_test.initializer)
        avg_acc = []
        while True:
            try:
                batch_test, label_test = sess.run(next_element_test)
                acc_test_val = sess.run(acc_test, feed_dict={x_input: batch_test, y_input: label_test})
                avg_acc.append(acc_test_val)
            except tf.errors.OutOfRangeError:
                print('end test ', np.sum(avg_acc)/len(y_test))
                break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--batch_size', default=64, type=int, required=True)
    parser_train.add_argument('--epoch', default=200, type=int, required=True)
    parser_train.add_argument('--network', default='resnet18', required=True)
    parser_train.add_argument('--opt', default='momentum', required=True)
    parser_train.add_argument('--train_path', default='/data/ChuyuanXiong/up/cifar-100-python/train', required=True)
    parser_train.add_argument('--test_path', default='/data/ChuyuanXiong/up/cifar-100-python/test', required=True)
    parser_train.set_defaults(func=train)

    # Test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--network', default='resnet18', required=True)
    parser_test.add_argument('--test_path', default='/data/ChuyuanXiong/up/cifar-100-python/test', required=True)
    parser_test.add_argument('--ckpt', default='params/resnet18/Speaker_vox_iter_58000.ckpt', required=True)
    parser_test.set_defaults(func=test)


    opt = parser.parse_args()
    opt.func(opt)





