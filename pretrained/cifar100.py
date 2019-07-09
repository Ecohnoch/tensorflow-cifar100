# import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import argparse
import random
import time
import math
import cv2
import os

from model.resnet34 import resnet18, resnet34
from model.resnet50 import resnet50, resnet110, resnet152
from model.serenset50 import se_resnet50, se_resnet110, se_resnet152
from model.densenet import densenet121, densenet161, densenet169, densenet201, densenet100bc, densenet190bc
from model.resnext import resnext50, resnext110, resnext152
from model.seresnext import se_resnext50, se_resnext110, se_resnext152

from model.seresnet_fixed import get_resnet

from utils import compute_mean_var, norm_images, unpickle, generate_tfrecord, norm_images_using_mean_var, lr_schedule_200ep, lr_schedule_300ep
from exceptions import InvalidNetworkName, InvalidTestSetPath, InvalidPretrainedModelPath




class cifar100(object):
    def __init__(self, model: str, pretrained_model=None):
        '''
        model: str, point out which CNN model to use.
        pretrained_model: if None, use default ckpt file to load CNN params.
            if not None, use ur ckpt file to load params.
        '''
        self.__model = model
        print('CIFAR100 (test:1/7): model load: ', self.__model)
        self.__pretrained_model = pretrained_model

        x_test, y_test = self.__load_data('cifar-100-python/test')
        print('CIFAR100 (test:2/7): x_test, y_test load.')
        self.__generate_tfrecord(x_test, y_test)
        print('CIFAR100 (test:3/7): tfrecords generated.')
        self.__build_graph()
        print('CIFAR100 (test:4/7): graph build.')

    def test(self):
        if self.__pretrained_model is not None:
            ckpt_file = self.__pretrained_model
        else:
            ckpt_file = self.get_pretrained_model(self.__model)
        print('CIFAR100 (test:5/7): ckpt_file_path: ', ckpt_file)
        ans = 0

        dataset_test = tf.data.TFRecordDataset('trans/test.tfrecords')
        dataset_test = dataset_test.map(self.parse_test)
        dataset_test = dataset_test.shuffle(buffer_size=10000)
        dataset_test = dataset_test.batch(128)
        iterator_test = dataset_test.make_initializable_iterator()
        next_element_test = iterator_test.get_next() 
        print('CIFAR100 (test:6/7): data iterator load.')

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, ckpt_file)
            sess.run(iterator_test.initializer)
            avg_acc = []
            while True:
                try:
                    batch_test, label_test = sess.run(next_element_test)
                    acc_test_val = sess.run(self.__acc_test, feed_dict={self.__x_input: batch_test, self.__y_input: label_test})
                    avg_acc.append(acc_test_val)
                except tf.errors.OutOfRangeError:
                    ans = np.mean(avg_acc)
                    print('CIFA100 (test:7/7): end test, acc:', ans)
        return ans
        

    def __load_data(self, test_data_path):
        if not os.path.exists(test_data_path):
            raise InvalidTestSetPath('invalid test data path: {}'.format(test_data_path))
        test = unpickle(test_data_path)
        test_data  = test[b'data']
        x_test = test_data.reshape(test_data.shape[0], 3, 32, 32)
        x_test = x_test.transpose(0, 2, 3, 1)
        x_test = norm_images(x_test)
        y_test= test[b'fine_labels']
        return x_test, y_test
    
    def __build_graph(self):
        self.__x_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.__y_input = tf.placeholder(tf.int64, [None, ])
        self.__y_input_one_hot = tf.one_hot(self.__y_input, 100)
        self.__prob_test = self.get_model(self.__x_input, self.__model)
        self.__logit_softmax_test = tf.nn.softmax(self.__prob_test)
        self.__acc_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.__logit_softmax_test, 1), self.__y_input), tf.float32))
    
    @staticmethod
    def __generate_tfrecord(x_test, y_test):
        if not os.path.exists('trans/') and not os.path.exists('trans/test.tfrecords'):
            generate_tfrecord(x_test, y_test, 'trans/', 'test.tfrecords')

    @staticmethod
    def get_model(x_input, network):
        if network == 'resnet50':
            return resnet50(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'resnet18':
            return resnet18(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'resnet34':
            return resnet34(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'seresnet50':
            return se_resnet50(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'resnet110':
            return resnet110(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'seresnet110':
            return se_resnet110(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'seresnet152':
            return se_resnet152(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'resnet152':
            return resnet152(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'seresnet_fixed':
            return get_resnet(x_input, 152, type='se_ir', trainable=False, reuse=True)
        elif network == 'densenet121':
            return densenet121(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'densenet169':
            return densenet169(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'densenet201':
            return densenet201(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'densenet161':
            return densenet161(x_input, is_training=False, reuse=False, kernel_initializer=None)
        elif network == 'densenet100bc':
            return densenet100bc(x_input, reuse=True, is_training=False, kernel_initializer=None)
        elif network == 'densenet190bc':
            return densenet190bc(x_input, reuse=True, is_training=False, kernel_initializer=None)
        elif network == 'resnext50':
            return resnext50(x_input, is_training=False, reuse=False, cardinality=32, kernel_initializer=None)
        elif network == 'resnext110':
            return resnext110(x_input, is_training=False, reuse=False, cardinality=32, kernel_initializer=None)
        elif network == 'resnext152':
            return resnext152(x_input, is_training=False, reuse=False, cardinality=32, kernel_initializer=None)
        elif network == 'seresnext50':
            return se_resnext50(x_input, reuse=True, is_training=False, cardinality=32, kernel_initializer=None)
        elif network == 'seresnext110':
            return se_resnext110(x_input, reuse=True, is_training=False, cardinality=32, kernel_initializer=None)
        elif network == 'seresnext152':
            return se_resnext152(x_input, reuse=True, is_training=False, cardinality=32, kernel_initializer=None)
        raise InvalidNetworkName('Network name is invalid!')
    
    @staticmethod
    def get_pretrained_model(network):
        if network == 'resnet18':
            return 'params/resnet18/Speaker_vox_iter_58000.ckpt'
        raise InvalidPretrainedModelPath('This network dosen\'t have default params yet.')
    
    @staticmethod
    def parse_test(example_proto):
        features = {'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)}
        features = tf.parse_single_example(example_proto, features)
        img = tf.decode_raw(features['image_raw'], tf.float32)
        img = tf.reshape(img, shape=(32, 32, 3))

        label = tf.cast(features['label'], tf.int64)
        return img, label
