import tensorflow as tf 
import numpy as np 

class bottleneck:
    def __init__(self, input_tensor, growth_rate, is_training=True, reuse=False, name='Default', kernel_initializer=None):
        self.growth_rate = growth_rate
        self.inner_channel = 4 * growth_rate

        self.is_training = is_training
        self.reuse = reuse
        self.name = name
        self.kernel_initializer = kernel_initializer

        self.input_tensor = input_tensor
        
    def bottle_neck(self):
        x = tf.layers.batch_normalization(self.input_tensor, training=self.is_training, reuse=self.reuse, name=self.name + 'bn0')
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, self.inner_channel, (1,1), use_bias=False, name=self.name+'conv1', reuse=self.reuse, kernel_initializer=self.kernel_initializer)
        x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, name=self.name + 'bn1')
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, self.growth_rate, (3,3), use_bias=False, padding='SAME', name=self.name+'conv2', reuse=self.reuse, kernel_initializer=self.kernel_initializer)
        # return tf.concat([x, self.input_tensor], axis=1)
        print('Bottle_neck_name: ', self.name)

        return tf.concat([x, self.input_tensor], 3)

class transition:
    def __init__(self, input_tensor, out_channels, is_training=True, reuse=False, name='Default', kernel_initializer=None):
        self.input_tensor = input_tensor
        self.out_channels = out_channels
        self.name = name

        self.is_training = is_training
        self.reuse = reuse
        self.kernel_initializer = kernel_initializer

    def down_sample(self):
        x = tf.layers.batch_normalization(self.input_tensor, training=self.is_training, reuse=self.reuse, name=self.name + 'bn0')
        x = tf.layers.conv2d(x, self.out_channels, (1,1), use_bias=False, name=self.name+'conv1', reuse=self.reuse, kernel_initializer=self.kernel_initializer)
        x = tf.layers.average_pooling2d(x, pool_size=[2,2], strides=[2,2], name=self.name+'avg_pool1')
        print('Transition: ', self.name)
        return x

class Desnet:
    def __init__(self, input_tensor, block, nblocks, growth_rate, reduction=0.5, n_class=100, is_training=True, reuse=False, kernel_initializer=None):
        self.inner_channel = 2 * growth_rate
        self.input_tensor = input_tensor

        self.block = block
        self.nblocks = nblocks
        self.growth_rate = growth_rate
        self.reduction = reduction
        self.n_class = n_class

        self.is_training = is_training
        self.reuse = reuse

        x = tf.layers.conv2d(self.input_tensor, self.inner_channel, (3,3), padding='SAME', use_bias=False, name='conv_first', kernel_initializer=kernel_initializer)
        
        for index in range(len(nblocks) - 1):
            print('make_layer_%d:'%(index), x)
            x = self.make_dense_layer(x, block, nblocks[index], name='block_'+str(index), kernel_initializer=kernel_initializer)
            self.inner_channel += growth_rate * nblocks[index]
            out_channels = int(self.reduction * self.inner_channel)
            x = transition(x, out_channels, is_training=self.is_training, reuse=self.reuse, name='trainsition_'+str(index), kernel_initializer=kernel_initializer).down_sample()
        
        x = self.make_dense_layer(x, block, nblocks[len(nblocks) - 1], name='last_block',kernel_initializer=kernel_initializer)
        x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, name='bn-1')
        x = tf.nn.relu(x)
        print('before gap:', x)
        x = tf.reduce_mean(x, [1, 2], name='gap')
        print('after gap:', x)
        x = tf.layers.dense(x, n_class, name='dense', reuse=reuse, kernel_initializer=kernel_initializer)
        self.output = x


    def make_dense_layer(self, x, block, nblocks, name='Default', kernel_initializer=None):
        for index in range(nblocks):
            obj = self.block(x, self.growth_rate, is_training=self.is_training, reuse=self.reuse, name=name+'blocks_'+str(index), kernel_initializer=kernel_initializer)
            x = obj.bottle_neck()
        return x

def densenet121(input_tensor, is_training, reuse, kernel_initializer=None):
    return Desnet(input_tensor, bottleneck, [6, 12, 24, 16], 32, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer).output

def densenet169(input_tensor, is_training, reuse, kernel_initializer=None):
    return Desnet(input_tensor, bottleneck, [6, 12, 32, 32], 32, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer).output

def densenet201(input_tensor, is_training, reuse, kernel_initializer=None):
    return Desnet(input_tensor, bottleneck, [6, 12, 48, 32], 32, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer).output

def densenet161(input_tensor, is_training, reuse, kernel_initializer=None):
    return Desnet(input_tensor, bottleneck, [6, 12, 36, 24], 48, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer).output
        




if __name__ == '__main__':
    a = np.random.rand(3, 32, 32, 3)
    inp = tf.placeholder(tf.float32, [None, 32, 32, 3])
    out = densenet201(inp, is_training=True, reuse=False, kernel_initializer=None)
    print(out)


