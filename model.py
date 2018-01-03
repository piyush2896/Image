import tensorflow as tf
from util import Config


USE_RELU = 0
USE_LEAKY_RELU = 1


def _conv2d(A, W, strides=[2, 2], padding='VALID'):
    strides = [1, strides[0], strides[1], 1]
    return tf.nn.conv2d(A, W, padding=padding, strides=strides)

def conv2d(A, k_size, strides=[2, 2], padding='VALID', name='conv'):
    W = tf.get_variable(name + '_W', shape=k_size)
    b = tf.Variable(tf.zeros((1, 1, 1, k_size[3])), name=name+'_b')
    Z = tf.add(_conv2d(A, W, padding=padding, strides=strides), b, name=name)

    params = {
        'W': W,
        'b': b
    }

    return Z, params

def flatten(tensor):
    return tf.contrib.layers.flatten(tensor)

def dense_layer(A, in_prev, out, name):
    W = tf.get_variable(name + '_W', shape=(in_prev, out))
    b = tf.Variable(tf.zeros((1, out)), name=name+'_b')
    Z = tf.add(tf.matmul(A, W), b, name=name)

    params = {
        'W': W,
        'b': b
    }

    return Z, params

def load_model(activation_type=USE_RELU, alpha=None):
    input_shape = [None] + Config.IMAGE_SHAPE
    params = {}
    model = {}

    if activation_type == USE_RELU:
        relu = tf.nn.relu
    else:
        # Custom Activation Function- Leaky Relu.
        if alpha == None:
            alpha = 0.05
        relu = lambda x: tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    model['input'] = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input')

    model['conv_1'], params['conv_1'] = conv2d(model['input'], k_size=[5, 5, 3, 64], name='conv_1')
    model['relu_1'] = relu(model['conv_1'])

    model['conv_2'], params['conv_2'] = conv2d(model['relu_1'], k_size=[5, 5, 64, 128], name='conv_2')
    model['relu_2'] = relu(model['conv_2'])

    model['pool_1'] = tf.nn.max_pool(model['relu_2'], ksize=(1, 5, 5, 1),
                                     strides=(1, 5, 5, 1), padding='VALID', name='pool_1')

    model['flatten'] = flatten(model['pool_1'])

    model['dense_1'], params['dense_1'] = dense_layer(model['flatten'], 
                                                      model['flatten'].get_shape()[-1],
                                                      256, 'dense_1')
    model['relu_3'] = relu(model['dense_1'])

    model['out'], params['out'] = dense_layer(model['dense_1'], 256, 1, 'out')

    return model, params
    

if __name__ == '__main__':
    from pprint import pprint
    model, params = load_model(activation_type=USE_LEAKY_RELU)
    pprint(model, indent=2)
