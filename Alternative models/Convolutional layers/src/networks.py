import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from .ops import conv2d, conv2d_transpose, pixelwise_accuracy

# example of using the vgg16 model as a feature extraction model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import time
from keras import Sequential
from tensorflow.keras import layers
from keras import backend as K
import keras
#import tensorflow as tf



class Discriminator(object):
    def __init__(self, name, kernels, training=True):
        self.name = name
        self.kernels = kernels
        self.training = training
        self.var_list = []

    def create(self, inputs, kernel_size=None, seed=None, reuse_variables=None):
        output = inputs
        with tf.compat.v1.variable_scope(self.name, reuse=reuse_variables):
            for index, kernel in enumerate(self.kernels):

                # not use batch-norm in the first layer
                bnorm = False if index == 0 else True
                name = 'conv' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    bnorm=bnorm,
                    activation=tf.compat.v1.nn.leaky_relu,
                    seed=seed
                )

                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.compat.v1.nn.dropout(output, keep_prob=keep_prob, name='dropout_' + name, seed=seed)
            
            #output = keras.layers.Dense(250, activation='relu')(output)
            #output = keras.layers.Dense(250, activation='relu')(output)

            output = conv2d(
                inputs=output,
                name='conv_last',
                filters=1,
                kernel_size=4,                  # last layer kernel size = 4
                strides=1,                      # last layer stride = 1
                bnorm=False,                    # do not use batch-norm for the last layer
                seed=seed
            )

            self.var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output


class Generator(object):
    def __init__(self, name, encoder_kernels, decoder_kernels, output_channels=3, training=True):
        self.name = name
        self.encoder_kernels = encoder_kernels
        self.decoder_kernels = decoder_kernels
        self.output_channels = output_channels
        self.training = training
        self.var_list = []
        #self.model = InceptionResNetV2()

    def create(self, inputs, kernel_size=None, seed=None, reuse_variables=None):
        output = inputs

        with tf.compat.v1.variable_scope(self.name, reuse=reuse_variables):

            layers = []

            # encoder branch
            for index, kernel in enumerate(self.encoder_kernels):

                name = 'conv' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.compat.v1.nn.leaky_relu,
                    seed=seed
                )

                # save contracting path layers to be used for skip connections
                layers.append(output)
                
                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.compat.v1.nn.dropout(output, keep_prob=keep_prob, name='dropout_' + name, seed=seed)

            #output = keras.layers.Dense(250, activation='relu')(output)
            #output = keras.layers.Dense(250, activation='relu')(output)

            """
            image = inputs
            image = preprocess_input(image)
            #model = InceptionResNetV2()
            # remove the output layer
            model = Model(inputs=image, outputs=self.model.layers[-2].output)
            # get extracted features
            print(image)
            features = model.predict(image)
            
            x = tf.constant(features)
            x2 = tf.expand_dims(x, 0)
            newfeat = tf.tile(x2, [2, 2, 1])
            output = tf.concat(newfeat, layers[-1])
            output = conv2d(inputs=output,name="ResShape",kernel_size=1,filters=512,strides=1,activation=tf.nn.leaky_relu,
                    seed=seed)
            time.sleep(10)
            """

            # decoder branch
            for index, kernel in enumerate(self.decoder_kernels):

                name = 'deconv' + str(index)
                output = conv2d_transpose(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.compat.v1.nn.relu,
                    seed=seed
                )

                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.compat.v1.nn.dropout(output, keep_prob=keep_prob, name='dropout_' + name, seed=seed)

                # concat the layer from the contracting path with the output of the current layer
                # concat only the channels (axis=3)
                output = tf.compat.v1.concat([layers[len(layers) - index - 2], output], axis=3)

            output = conv2d(
                inputs=output,
                name='conv_last',
                filters=self.output_channels,   # number of output chanels
                kernel_size=1,                  # last layer kernel size = 1
                strides=1,                      # last layer stride = 1
                bnorm=False,                    # do not use batch-norm for the last layer
                activation=tf.compat.v1.nn.tanh,          # tanh activation function for the output
                seed=seed
            )

            self.var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output
