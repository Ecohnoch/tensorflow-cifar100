import tensorflow as tf 
import tensorlayer as tl
import numpy as np 

def conv2d(input_tensor, filters, kernel_size, strides, act=None, padding='SAME', w_init=None, b_init=None, name=None):
    return tl.layers.Conv2d(input_tensor, n_filter=filters, filter_size=kernel_size, strides=strides, act=act, padding=padding, W_init=w_init, b_init=b_init, name=name)

def bn(input_tensor, is_train, name, act=None):
    return tl.layers.BatchNormLayer(input_tensor, is_train=is_train, name=name, act=act)

def prelu(input_tensor):
	return tl.layers.PReluLayer(input_tensor)

def identity_block2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, kernel_initializer=None):
	filters1, filters2, filters3 = filters

	conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
	bn_name_1   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_reduce'

	x = conv2d(input_tensor, filters1, (1, 1), strides=(1,1), padding='SAME', name=conv_name_1, w_init=kernel_initializer)
	x = bn(x, is_train=is_training, name=bn_name_1, act=tf.nn.relu)

	conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
	bn_name_2   = 'bn'   + str(stage) + '_' + str(block) + '_3x3'
	x = conv2d(x, filters2, kernel_size=(3,3), strides=(1,1), padding='SAME',  name=conv_name_2, w_init=kernel_initializer)
	x = bn(x, is_train=is_training, name=bn_name_2, act=tf.nn.relu)

	conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
	bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
	x = conv2d(x, filters3, (1,1), strides=(1,1), name=conv_name_3, padding='SAME', w_init=kernel_initializer)
	x = bn(x, is_train=is_training, name=bn_name_3, act=None)
	
	squeeze = tl.layers.InputLayer(tf.reduce_mean(x.outputs, axis=[1, 2]), name=str(stage)+str(block)+'sq-gap')
	excitation1 = tl.layers.DenseLayer(squeeze, n_units=int(filters3/16.0), act=tf.nn.relu, W_init=kernel_initializer, name=str(stage)+str(block)+'excitation1')
	excitation2 = tl.layers.DenseLayer(excitation1, n_units=filters3, act=tf.nn.sigmoid, W_init=kernel_initializer, name=str(stage)+str(block)+'excitation2')
	scale = tl.layers.ReshapeLayer(excitation2, shape=[tf.shape(excitation2.outputs)[0], 1, 1, filters3], name=str(stage)+str(block)+'scale')

	residual_se = tl.layers.ElementwiseLayer([x, scale], combine_fn=tf.multiply, name=str(stage)+str(block)+'scale_layer', act=None)
	x = tl.layers.ElementwiseLayer([residual_se, input_tensor], combine_fn=tf.add, act=tf.nn.relu, name=str(stage)+str(block)+'elementwise')
	return x


def conv_block_2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(2, 2), kernel_initializer=None):
	filters1, filters2, filters3 = filters

	conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
	bn_name_1   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_reduce'
	x = conv2d(input_tensor, filters1, (1, 1), strides=strides, padding='SAME', name=conv_name_1, w_init=kernel_initializer)
	x = bn(x, is_train=is_training, name=bn_name_1, act=tf.nn.relu)

	conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
	bn_name_2   = 'bn'   + str(stage) + '_' + str(block) + '_3x3'
	x = conv2d(x, filters2, kernel_size=(3,3), strides=(1,1), padding='SAME', name=conv_name_2, w_init=kernel_initializer)
	x = bn(x, is_train=is_training, name=bn_name_2, act=tf.nn.relu)

	conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
	bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
	x = conv2d(x, filters3, (1,1), strides=(1,1), name=conv_name_3, padding='SAME', w_init=kernel_initializer)
	x = bn(x, is_train=is_training, name=bn_name_3, act=tf.nn.relu)


	squeeze = tl.layers.InputLayer(tf.reduce_mean(x.outputs, axis=[1, 2]), name=str(stage)+str(block)+'sq-gap')
	excitation1 = tl.layers.DenseLayer(squeeze, n_units=int(filters3/16.0), act=tf.nn.relu, W_init=kernel_initializer, name=str(stage)+str(block)+'excitation1')
	excitation2 = tl.layers.DenseLayer(excitation1, n_units=filters3, act=tf.nn.sigmoid, W_init=kernel_initializer, name=str(stage)+str(block)+'excitation2')
	scale = tl.layers.ReshapeLayer(excitation2, shape=[tf.shape(excitation2.outputs)[0], 1, 1, filters3], name=str(stage)+str(block)+'scale')

	residual_se = tl.layers.ElementwiseLayer([x, scale], combine_fn=tf.multiply, name=str(stage)+str(block)+'scale_layer', act=None)

	conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
	bn_name_4   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_shortcut'
	shortcut = conv2d(input_tensor, filters3, (1,1), strides=strides, padding='SAME', name=conv_name_4, w_init=kernel_initializer)
	shortcut = bn(shortcut, is_train=is_training, name=bn_name_4)

	x = tl.layers.ElementwiseLayer([residual_se, shortcut], combine_fn=tf.add, act=tf.nn.relu, name=str(stage)+str(block)+'elementwise')
	return x


def get_se_resnet(input_tensor, block, is_training, reuse, kernel_initializer=None):
	# 3, 4, 16, 3
	with tf.variable_scope('scope', reuse=reuse):
		x = tl.layers.InputLayer(input_tensor, name='inputs')
		x = conv2d(x, 64, (3,3), strides=(1,1), padding='SAME', w_init=kernel_initializer, name='face_conv1_1/3x3_s1')
		x = bn(x, is_train=is_training, name='face_bn1_1/3x3_s1', act=tf.nn.relu)

		x = conv_block_2d(x, 3, [64, 64, 256], stage=2, block='face_1a', strides=(1,1), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
		for first_block in range(block[0] - 1):
			x = identity_block2d(x, 3, [64, 64, 256], stage='1b_{}'.format(first_block), block='face_{}'.format(first_block), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

		x = conv_block_2d(x, 3, [128, 128, 512], stage=3, block='face_2a', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
		for second_block in range(block[1] - 1):
			x = identity_block2d(x, 3, [128, 128, 512], stage='2b_{}'.format(second_block), block='face_{}'.format(second_block), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

		x = conv_block_2d(x, 3, [256, 256, 1024], stage=4, block='face_3a' , is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
		for third_block in range(block[2] - 1):
			x = identity_block2d(x, 3, [256, 256, 1024], stage='3b_{}'.format(third_block), block='face_{}'.format(third_block), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
		
		x = conv_block_2d(x, 3, [512, 512, 2048], stage=5, block='face_4a', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
		for fourth_block in range(block[3] - 1):
			x = identity_block2d(x, 3, [512, 512, 2048], stage='4b_{}'.format(fourth_block), block='face_{}'.format(fourth_block), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
		
		# pooling_output = tf.layers.max_pooling2d(x4, (7,7), strides=(1,1), name='mpool2')
		# print('before gap: ', x)
		
		pooling_output = tl.layers.GlobalMeanPool2d(x, name='gap')
		fc_output      = tl.layers.DenseLayer(pooling_output, 100, name='face_fc1', W_init=tf.contrib.layers.xavier_initializer(), b_init=tf.zeros_initializer())

	return fc_output.outputs

def se_resnet50(input_tensor, is_training, reuse, kernel_initializer=None):
	return get_se_resnet(input_tensor, [3,4,6,3], is_training, reuse, kernel_initializer)

def se_resnet110(input_tensor, is_training, reuse, kernel_initializer=None):
	return get_se_resnet(input_tensor, [3,4,23,3], is_training, reuse, kernel_initializer)

def se_resnet152(input_tensor, is_training, reuse, kernel_initializer=None):
	return get_se_resnet(input_tensor, [3,8,36,3], is_training, reuse, kernel_initializer)

if __name__ == '__main__':
	example_data = [np.random.rand(32, 32, 3)]
	x = tf.placeholder(tf.float32, [None, 32, 32, 3])
	y = se_resnet50(x, is_training=True, reuse=False)
	print(y)
