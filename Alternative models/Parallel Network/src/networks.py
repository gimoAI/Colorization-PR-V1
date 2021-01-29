import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from .ops import conv2d, conv2d_transpose, pixelwise_accuracy


class Discriminator(object):
    def __init__(self, name, kernels, training=True):
        self.name = name
        self.kernels = kernels
        self.training = training
        self.var_list = []

    def create3(self, inputs, kernel_size=3, seed=None, reuse_variables=None):
        output = inputs
        with tf.variable_scope('3convv_dis', reuse=reuse_variables):
            for index, kernel in enumerate(self.kernels):

                # not use batch-norm in the first layer
                bnorm = False if index == 0 else True
                name = '3conv_dis' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    bnorm=bnorm,
                    activation=tf.nn.leaky_relu,
                    seed=seed
                )

                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.nn.dropout(output, keep_prob=keep_prob, name='dropout_' + name, seed=seed)
                    
            return output

                    
    def create5(self, inputs, kernel_size=5, seed=None, reuse_variables=None):
        output = inputs
        with tf.variable_scope('5convv_dis', reuse=reuse_variables):
            for index, kernel in enumerate(self.kernels):

                # not use batch-norm in the first layer
                bnorm = False if index == 0 else True
                name = '5conv_dis' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    bnorm=bnorm,
                    activation=tf.nn.leaky_relu,
                    seed=seed
                )

                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.nn.dropout(output, keep_prob=keep_prob, name='dropout_' + name, seed=seed)
            
            return output

    def create(self, inputs, kernel_size=None, seed=None, reuse_variables=None):
        
        with tf.variable_scope(self.name, reuse=reuse_variables):

            output = conv2d(
                inputs=tf.concat([self.create5(inputs = inputs, kernel_size = 5, seed =  seed, reuse_variables= reuse_variables),
                        self.create3(inputs = inputs, kernel_size = 3, seed =  seed, reuse_variables= reuse_variables)], axis = 3),
                name='conv_last_dis',
                filters=1,
                kernel_size=4,                  # last layer kernel size = 4
                strides=1,                      # last layer stride = 1
                bnorm=False,                    # do not use batch-norm for the last layer
                seed=seed
            )

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output


class Generator(object):
    def __init__(self, name, encoder_kernels, decoder_kernels, output_channels=3, training=True):
        self.name = name
        self.encoder_kernels = encoder_kernels
        self.decoder_kernels = decoder_kernels
        self.output_channels = output_channels
        self.training = training
        self.var_list = []

    def create5(self, inputs, kernel_size=5, seed=None, reuse_variables=None):
        output = inputs

        with tf.variable_scope('4convv', reuse=reuse_variables):

            layers = []

            # encoder branch
            for index, kernel in enumerate(self.encoder_kernels):

                name = '5conv' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.nn.leaky_relu,
                    seed=seed
                )

                # save contracting path layers to be used for skip connections
                layers.append(output)
                
                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.nn.dropout(output, keep_prob=keep_prob, name='4dropout_' + name, seed=seed)

            # decoder branch
            for index, kernel in enumerate(self.decoder_kernels):

                name = '5deconv' + str(index)
                output = conv2d_transpose(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.nn.relu,
                    seed=seed
                )

                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.nn.dropout(output, keep_prob=keep_prob, name='4dropout_' + name, seed=seed)

                # concat the layer from the contracting path with the output of the current layer
                # concat only the channels (axis=3)
                output = tf.concat([layers[len(layers) - index - 2], output], axis=3)
            
            return output
        
    def create3(self, inputs, kernel_size=3, seed=None, reuse_variables=None):
        output = inputs

        with tf.variable_scope('3convv', reuse=reuse_variables):

            layers = []

            # encoder branch
            for index, kernel in enumerate(self.encoder_kernels):

                name = '3conv' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.nn.leaky_relu,
                    seed=seed
                )

                # save contracting path layers to be used for skip connections
                layers.append(output)
                
                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.nn.dropout(output, keep_prob=keep_prob, name='3dropout_' + name, seed=seed)

            # decoder branch
            for index, kernel in enumerate(self.decoder_kernels):

                name = '3deconv' + str(index)
                output = conv2d_transpose(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.nn.relu,
                    seed=seed
                )

                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.nn.dropout(output, keep_prob=keep_prob, name='3dropout_' + name, seed=seed)

                # concat the layer from the contracting path with the output of the current layer
                # concat only the channels (axis=3)
                output = tf.concat([layers[len(layers) - index - 2], output], axis=3)
            
            return output


    def create(self, inputs, kernel_size=None, seed=None, reuse_variables=None):
        
        with tf.variable_scope(self.name, reuse=reuse_variables):
            

            output = conv2d(
                inputs=tf.concat([self.create5(inputs = inputs, kernel_size = 4, seed =  seed, reuse_variables= reuse_variables),
                        self.create3(inputs = inputs, kernel_size = 3, seed =  seed, reuse_variables= reuse_variables)], axis = 3),
                name='conv_last',
                filters=self.output_channels,   # number of output chanels
                kernel_size=1,                  # last layer kernel size = 1
                strides=1,                      # last layer stride = 1
                bnorm=False,                    # do not use batch-norm for the last layer
                activation=tf.nn.tanh,          # tanh activation function for the output
                seed=seed
            )

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output
