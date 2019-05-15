import numpy as np 
import tensorflow as tf
import random
import pickle
import os

def compute_mean_var(image):
    # image.shape: [image_num, w, h, c]
    mean = []
    var  = []
    for c in range(image.shape[-1]):
        mean.append(np.mean(image[:, :, :, c]))
        var.append(np.std(image[:, :, :, c]))
    return mean, var

def norm_images(image):
    # image.shape: [image_num, w, h, c]
    image = image.astype('float32')
    mean, var = compute_mean_var(image)
    image[:, :, :, 0] = (image[:, :, :, 0] - mean[0]) / var[0]
    image[:, :, :, 1] = (image[:, :, :, 1] - mean[1]) / var[1]
    image[:, :, :, 2] = (image[:, :, :, 2] - mean[2]) / var[2]
    return image

def norm_images_using_mean_var(image, mean, var):
    image = image.astype('float32')
    image[:, :, :, 0] = (image[:, :, :, 0] - mean[0]) / var[0]
    image[:, :, :, 1] = (image[:, :, :, 1] - mean[1]) / var[1]
    image[:, :, :, 2] = (image[:, :, :, 2] - mean[2]) / var[2]
    return image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def generate_tfrecord(train, labels, output_path, output_name):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    writer = tf.python_io.TFRecordWriter(os.path.join(output_path, output_name))
    for ind, (file, label) in enumerate(zip(train, labels)):
        img_raw = file.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if ind != 0 and ind % 1000 == 0:
            print("%d num imgs processed" % ind)
    writer.close()

