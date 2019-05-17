import tensorflow as tf 

def identity_block2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
	filters1, filters2, filters3 = filters

	conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
	bn_name_2   = 'bn'   + str(stage) + '_' + str(block) + '_3x3'
	x = tf.layers.conv2d(input_tensor, filters2, kernel_size, use_bias=False, padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
	bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
	x = tf.layers.conv2d(x, filters3, (kernel_size, kernel_size), use_bias=False,  padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)
	
	x = tf.add(input_tensor, x)
	x = tf.nn.relu(x)
	return x

def conv_block_2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(2, 2), kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
	filters1, filters2, filters3 = filters

	conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
	bn_name_2   = 'bn'   + str(stage) + '_' + str(block) + '_3x3'
	x = tf.layers.conv2d(input_tensor, filters2, (kernel_size, kernel_size), use_bias=False, strides=strides, padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
	bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
	x = tf.layers.conv2d(x, filters3, (kernel_size, kernel_size), use_bias=False, padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

	conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
	bn_name_4   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_shortcut'
	shortcut = tf.layers.conv2d(input_tensor, filters3, (kernel_size, kernel_size), use_bias=False, strides=strides, padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_4, reuse=reuse)
	shortcut = tf.layers.batch_normalization(shortcut, training=is_training, name=bn_name_4, reuse=reuse)

	x = tf.add(shortcut, x)
	x = tf.nn.relu(x)
	return x


def resnet18(input_tensor, is_training=True, pooling_and_fc=True, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
	x = tf.layers.conv2d(input_tensor, 64, (3,3), strides=(1,1), kernel_initializer=kernel_initializer, use_bias=False, padding='SAME', name='conv1_1/3x3_s1', reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name='bn1_1/3x3_s1', reuse=reuse)
	x = tf.nn.relu(x)

	x1 = identity_block2d(x, 3, [48, 64, 64], stage=2, block='1b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x1 = identity_block2d(x1, 3, [48, 64, 64], stage=3, block='1c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)


	x2 = conv_block_2d(x1, 3, [96, 128, 128], stage=3, block='2a', strides=(2,2), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x2 = identity_block2d(x2, 3, [96, 128, 128], stage=3, block='2b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

	x3 = conv_block_2d(x2, 3, [128, 256, 256], stage=4, block='3a', strides=(2,2), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [128, 256, 256], stage=4, block='3b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

	x4 = conv_block_2d(x3, 3, [256, 512, 512], stage=5, block='4a', strides=(2,2), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x4 = identity_block2d(x4, 3, [256, 512, 512], stage=5, block='4b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

	print('before gap: ', x4)
	x4 = tf.reduce_mean(x4, [1,2])
	print('after gap: ', x4)
	# flatten = tf.contrib.layers.flatten(x4)
	prob = tf.layers.dense(x4, 100, reuse=reuse, kernel_initializer=tf.contrib.layers.xavier_initializer())
	# prob = tf.layers.batch_normalization(prob, training=is_training, name='fbn', reuse=reuse)
	print('prob', prob)

	return prob

def resnet34(input_tensor, is_training=True, pooling_and_fc=True, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
	x = tf.layers.conv2d(input_tensor, 64, (3,3), strides=(1,1), kernel_initializer=kernel_initializer, use_bias=False, padding='SAME', name='conv1_1/3x3_s1', reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name='bn1_1/3x3_s1', reuse=reuse)
	x = tf.nn.relu(x)

	x1 = identity_block2d(x, 3, [48, 64, 64], stage=1, block='1a', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x1 = identity_block2d(x1, 3, [48, 64, 64], stage=1, block='1b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x1 = identity_block2d(x1, 3, [48, 64, 64], stage=1, block='1c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)


	x2 = conv_block_2d(x1, 3, [96, 128, 128], stage=2, block='2a', strides=(2,2), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x2 = identity_block2d(x2, 3, [96, 128, 128], stage=2, block='2b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x2 = identity_block2d(x2, 3, [96, 128, 128], stage=2, block='2c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x2 = identity_block2d(x2, 3, [96, 128, 128], stage=2, block='2d', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)


	x3 = conv_block_2d(x2, 3, [128, 256, 256], stage=3, block='3a', strides=(2,2), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [128, 256, 256], stage=3, block='3b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [128, 256, 256], stage=3, block='3c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [128, 256, 256], stage=3, block='3d', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [128, 256, 256], stage=3, block='3e', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [128, 256, 256], stage=3, block='3f', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)


	x4 = conv_block_2d(x3, 3, [256, 512, 512], stage=4, block='4a', strides=(2,2), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x4 = identity_block2d(x4, 3, [256, 512, 512], stage=4, block='4b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x4 = identity_block2d(x4, 3, [256, 512, 512], stage=4, block='4c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)


	print('before gap: ', x4)
	x4 = tf.reduce_mean(x4, [1,2])
	print('after gap: ', x4)
	# flatten = tf.contrib.layers.flatten(x4)
	prob = tf.layers.dense(x4, 100, reuse=reuse, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
	# prob = tf.layers.batch_normalization(prob, training=is_training, name='fbn', reuse=reuse)
	print('prob', prob)

	return prob