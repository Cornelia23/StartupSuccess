import tensorflow as tf
import numpy as np
import random
import math
import pandas as pd

class StartupCNN(tf.keras.Model):
    '''
        Initializes the cnn-based model for predicting startup success..
        
        -Inputs: categorical_inputs, numeric_inputs
        -Output: logits
    '''
    def __init__(self, vocab_sizes, num_numeric_features, embedding_dim=8, **kwargs):
       super().__init__(**kwargs)
       self.vocab_sizes = vocab_sizes
       self.num_numeric_features = num_numeric_features
       self.embedding_dim = embedding_dim
       self.embedding_layers = [tf.keras.layers.Embedding(
                input_dim = vocab_size + 1,
                output_dim = self.embedding_dim)
                for vocab_size in self.vocab_sizes]
       
       self.conv1 = tf.keras.layers.Conv1D(filters=64,kernel_size=3, activation = 'relu', padding = 'same')
       self.batch_norm1 = tf.keras.layers.BatchNormalization()
       self.conv2 = tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation = 'relu', padding='same')
       self.batch_norm2 = tf.keras.layers.BatchNormalization()
       self.flatten = tf.keras.layers.Flatten()
       self.hidden_layer = tf.keras.layers.Dense(64, activation= 'relu')
       self.dropout_layer = tf.keras.layers.Dropout(0.25)
       self.output_layer = tf.keras.layers.Dense(1, activation = 'softmax')

    def call(self, inputs, training=False):
        categorical_inputs = inputs['categorical_inputs']
        numeric_inputs = inputs.get('numeric_inputs', None)
        embeddings = []
        for i, embedding_layer in enumerate(self.embedding_layers):
            cat_feature = categorical_inputs[:,i:i+1]
            embedding = embedding_layer(cat_feature)
            embedding = tf.squeeze(embedding, axis=1)
            embeddings.append(embedding)

        cat_features = tf.concat(embeddings, axis=1)
        x = tf.concat([cat_features, numeric_inputs], axis=1)
        x = tf.expand_dims(x, axis=-1)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.flatten(x)
        x = self.hidden_layer(x)
        x = self.dropout_layer(x)
        logits = self.output_layer(x)
        return logits



        







