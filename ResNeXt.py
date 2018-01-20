import tensorpack.dataflow.dataset as dataset
import numpy as np
import tensorflow as tf
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train, test = dataset.Cifar10('train'), dataset.Cifar10('test')

n = 50000
x_train = np.array([train.data[i][0] for i in range(n)], dtype=np.float32)
y_train = np.array([train.data[i][1] for i in range(n)], dtype=np.int32)
x_test = np.array([ex[0] for ex in test.data], dtype=np.float32)
y_test = np.array([ex[1] for ex in test.data], dtype=np.int32)

del (train, test)  # frees approximately 180 MB

Y = np.zeros([n, 10])
for i in range(y_train.shape[0]):
    Y[i, y_train[i]] = 1
y_train = Y

Y = np.zeros([10000, 10])
for i in range(y_test.shape[0]):
    Y[i, y_test[i]] = 1
y_test = Y

# standardization
x_train_pixel_mean = x_train.mean(axis=0)  # per-pixel mean
x_train_pixel_std = x_train.std(axis=0)  # per-pixel std
x_train -= x_train_pixel_mean
x_train /= x_train_pixel_std
x_test -= x_train_pixel_mean
x_test /= x_train_pixel_std
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


weight_decay = 0.0005
momentum = 0.9
lr = 0.1

cardinality = 8  # 16
depth = 64
block_depth = 3

batch_size = 100
iteration = n // batch_size
test_iteration = 100
num_epochs = 10

def data_augmentation(X):
    for i in range(len(X)):
        if bool(random.getrandbits(1)):
            X[i] = np.fliplr(X[i])
    new_X = []
    pad = ((4, 4), (4, 4), (0, 0))  # both height, width with 4 pixels
    for i in range(len(X)):
        new_X.append(X[i])
        new_X[i] = np.lib.pad(X[i], pad_width=pad, mode='constant', constant_values=0)  #pad with 0
        h = random.randint(0, 8)
        w = random.randint(0, 8)
        new_X[i] = new_X[i][h:h + 32, w:w + 32]
    return new_X

def Batch_Normalization(x, training, scope):
    n_out = int(np.shape(x)[3])
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

class ResNeXt():
    def __init__(self, training):
        self.training = training

    def conv_1_3x3(self, x):
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], strides=1, padding='SAME', use_bias=False, name='conv_1_3x3')
        x = Batch_Normalization(x, training=self.training, scope='conv_1_3x3')
        x = tf.nn.relu(x)
        return x

    def ResNeXtBottleneck(self, input_x, out_dim, stride, name):
        layers_split = list()
        for i in range(cardinality):
            scope = name + '_cardinality_' + str(i)  # stage1_block_1_cardinality_1
            x = tf.layers.conv2d(inputs=input_x, filters=depth, kernel_size=[1, 1], strides=stride, padding='SAME', use_bias=False, name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_bn_reduce')
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(inputs=x, filters=depth, kernel_size=[3, 3], strides=1, padding='SAME', use_bias=False, name=scope+'_conv_conv')
            x = Batch_Normalization(x, training=self.training, scope=scope + '_bn_bn')
            x = tf.nn.relu(x)
            layers_split.append(x)

        x = tf.concat(layers_split, axis=3)
        x = tf.layers.conv2d(inputs=x, filters=out_dim, kernel_size=[1, 1], strides=1, padding='SAME', use_bias=False, name=name+'_conv_expand')
        x = Batch_Normalization(x, training=self.training, scope=name+'_bn_expand')
        return x

    def stage(self, input_x, in_dim, out_dim, stage_name, pool_stride):

        for i in range(block_depth):
            name = stage_name+'_block_'+str(i)  # stage1_block_1

            if i == 0:
                x = self.ResNeXtBottleneck(input_x, out_dim=out_dim, stride=pool_stride, name=name)
                if pool_stride == 2:
                    shortcut = tf.layers.average_pooling2d(inputs=input_x, pool_size=[2, 2], strides=2, padding='SAME')
                    shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [in_dim//2, in_dim//2]])
                else:
                    shortcut = input_x
                input_x = tf.nn.relu(x + shortcut)
            else:
                x = self.ResNeXtBottleneck(input_x, out_dim=out_dim, stride=1, name=name)
                shortcut = input_x
                input_x = tf.nn.relu(x + shortcut)
        return x

    def Forward(self, input_x):
        x = self.conv_1_3x3(input_x)
        x = self.stage(x, in_dim=64, out_dim=64, stage_name='stage1', pool_stride=1)
        x = self.stage(x, in_dim=64, out_dim=128, stage_name='stage2', pool_stride=2)
        x = self.stage(x, in_dim=128, out_dim=256, stage_name='stage3', pool_stride=2)
        x = tf.layers.average_pooling2d(x, 8, 1)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(inputs=x, use_bias=False, units=10, name='linear', reuse=None)
        return x

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32)

logits = ResNeXt(flag).Forward(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('done')
