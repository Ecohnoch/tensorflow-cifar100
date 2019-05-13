import tensorflow as tf 
import numpy as np 

def identity_block2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, kernel_initializer=None):
	filters1, filters2, filters3 = filters

	conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
	bn_name_1   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_reduce'

	x = tf.layers.conv2d(input_tensor, filters1, (1, 1), use_bias=False, padding='SAME', name=conv_name_1, reuse=reuse, kernel_initializer=kernel_initializer)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
	bn_name_2   = 'bn'   + str(stage) + '_' + str(block) + '_3x3'
	x = tf.layers.conv2d(x, filters2, kernel_size=(3,3), padding='SAME', use_bias=False, name=conv_name_2, reuse=reuse, kernel_initializer=kernel_initializer)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
	bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
	x = tf.layers.conv2d(x, filters3, (1,1), name=conv_name_3, padding='SAME', use_bias=False, reuse=reuse, kernel_initializer=kernel_initializer)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)
	
	x = tf.add(input_tensor, x)
	x = tf.nn.relu(x)
	return x


def conv_block_2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(2, 2), kernel_initializer=None):
	filters1, filters2, filters3 = filters

	conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
	bn_name_1   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_reduce'
	x = tf.layers.conv2d(input_tensor, filters1, (1, 1), use_bias=False, padding='SAME',  strides=strides, name=conv_name_1, reuse=reuse, kernel_initializer=kernel_initializer)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
	bn_name_2   = 'bn'   + str(stage) + '_' + str(block) + '_3x3'
	x = tf.layers.conv2d(x, filters2, kernel_size=(3,3), padding='SAME', use_bias=False, name=conv_name_2, reuse=reuse, kernel_initializer=kernel_initializer)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
	bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
	x = tf.layers.conv2d(x, filters3, (1,1), name=conv_name_3, padding='SAME', use_bias=False, reuse=reuse, kernel_initializer=kernel_initializer)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

	conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
	bn_name_4   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_shortcut'
	shortcut = tf.layers.conv2d(input_tensor, filters3, (1,1), padding='SAME', use_bias=False, strides=strides, name=conv_name_4, reuse=reuse, kernel_initializer=kernel_initializer)
	shortcut = tf.layers.batch_normalization(shortcut, training=is_training, name=bn_name_4, reuse=reuse)

	x = tf.add(shortcut, x)
	x = tf.nn.relu(x)
	return x


def resnet50(input_tensor, is_training=True, pooling_and_fc=True, reuse=False, kernel_initializer=tf.contrib.layers.xavier_initializer()):
	x = tf.layers.conv2d(input_tensor, 64, (3,3), strides=(1,1), padding='SAME', use_bias=False, kernel_initializer=kernel_initializer, name='face_conv1_1/3x3_s1', reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name='face_bn1_1/3x3_s1', reuse=reuse)
	x = tf.nn.relu(x)
	# x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), name='mpool1')

	x1 = conv_block_2d(x, 3, [64, 64, 256], stage=2, block='face_1a', strides=(1,1), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x1 = identity_block2d(x1, 3, [64, 64, 256], stage=2, block='face_1b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x1 = identity_block2d(x1, 3, [64, 64, 256], stage=2, block='face_1c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

	x2 = conv_block_2d(x1, 3, [128, 128, 512], stage=3, block='face_2a', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x2 = identity_block2d(x2, 3, [128, 128, 512], stage=3, block='face_2b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x2 = identity_block2d(x2, 3, [128, 128, 512], stage=3, block='face_2c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x2 = identity_block2d(x2, 3, [128, 128, 512], stage=3, block='face_2d', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

	x3 = conv_block_2d(x2, 3, [256, 256, 1024], stage=4, block='face_3a' , is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='face_3b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='face_3c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='face_3d', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='face_3e', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [256, 256, 1024], stage=4, block='face_3f', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)


	x4 = conv_block_2d(x3, 3, [512, 512, 2048], stage=5, block='face_4a', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x4 = identity_block2d(x4, 3, [512, 512, 2048], stage=5, block='face_4b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x4 = identity_block2d(x4, 3, [512, 512, 2048], stage=5, block='face_4c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

	if pooling_and_fc:
		# pooling_output = tf.layers.max_pooling2d(x4, (7,7), strides=(1,1), name='mpool2')
		print('before gap: ', x4)
		pooling_output = tf.reduce_mean(x4, [1, 2])
		fc_output      = tf.layers.dense(pooling_output, 100, name='face_fc1', reuse=reuse, kernel_initializer=tf.contrib.layers.xavier_initializer())
		# fc_output      = tf.layers.batch_normalization(fc_output, training=is_training, reuse=reuse, name='face_fbn')

	return fc_output

if __name__ == '__main__':
	example_data = [np.random.rand(112, 112, 3)]
	x = tf.placeholder(tf.float32, [None, 112, 112, 3])
	y = resnet50(x, is_training=True, reuse=False)
	print(y)

	with tf.Session() as sess:
		writer = tf.summary.FileWriter("logs/", sess.graph)
		init = tf.global_variables_initializer()
		sess.run(init)
