from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from manual_convolution import ManualConv2d
from base_model import CifarModel
from typing import List

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class CNN(CifarModel):
    def __init__(self, classes: List[int]) -> None:
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        
        :param classes: List of class indices (e.g., [3, 5] for cat and dog from CIFAR-10)
        """
        super(CNN, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.image_channels = 3
        self.num_classes = len(classes)
        self.hidden_layer_size: int = 320  # size of dense layers

        self.epsilon: float = 1e-3  # used for batch normalization only!
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), 1, activation="relu", input_shape=(self.input_width, self.input_height, self.image_channels))
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), 1, activation="relu", input_shape=(self.input_width, self.input_height, self.image_channels))
        self.pooling_layer = tf.keras.layers.MaxPooling2D(2,2)
        self.manualconv = None

        self.batch_1 = tf.keras.layers.BatchNormalization(axis = -1)
        self.batch_2 = tf.keras.layers.BatchNormalization(axis = -1)

        self.flatten = tf.keras.layers.Flatten()

        self.dense_hidden = tf.keras.layers.Dense(self.hidden_layer_size, activation = "relu")
        self.dense_hidden_2 = tf.keras.layers.Dense(self.hidden_layer_size // 2, activation = "relu")
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation = "softmax")
    

    def call(self, inputs: tf.Tensor, is_testing: bool = False) -> tf.Tensor:
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)



        conv1 = self.conv1(inputs)
        batch1 = self.batch_1(conv1)
        pool1 = self.pooling_layer(batch1)
        
        if is_testing is True:
            if self.manualconv is None:
                self.manualconv = ManualConv2d((3,3), [1,1,1,1], "VALID", True, False)

                if len(self.conv2.weights) >= 2:
                    self.manualconv.set_weights(self.conv2.weights[0], self.conv2.weights[1])
                else:
                    self.manualconv.set_weights(self.conv2.weights[0], None)
            conv2 = self.manualconv(pool1)
            conv2 = tf.nn.relu(conv2)
            
        
        else:
            conv2 = self.conv2(pool1)
        
        batch2 = self.batch_2(conv2)
        pool2 = self.pooling_layer(batch2)

        flattened_data = self.flatten(pool2)
        dense_hidden = self.dense_hidden(flattened_data)
        dropout_1 = tf.nn.dropout(dense_hidden, rate = 0.25)
        dense_hidden_2 = self.dense_hidden_2(dropout_1)
        dropout_2 = tf.nn.dropout(dense_hidden_2, rate = 0.25)
        output = self.output_layer(dropout_2)

        return output
