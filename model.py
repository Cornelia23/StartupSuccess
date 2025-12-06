import tensorflow as tf 
import numpy as np
from transformer import TransformerLanguageModel, LanguageTransformerBlock, MultiHeadAttention, PositionalEncoding
from cnn import CNN

class TransformerCNNModel(tf.keras.Model):
    '''Model must incorporate categorical data and train with transformer and fuse data with the quantitative data trained on the CNN. '''
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=None,
                 max_seq_length=512, dropout_rate=0.1, pad_token_id=0, use_cnn=True, **kwargs):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.pad_token_id = pad_token_id
        self.use_cnn = use_cnn
        num_output_classes = 4 #Number of outcomes

        self.transformer = TransformerLanguageModel(
            vocab_size=self.vocab_size,
            d_model = self.d_model,
            n_heads = self.n_heads,
            n_layers = self.n_layers,
            d_ff = self.d_ff,
        )

        if self.use_cnn: 
            self.cnn = CNN([0,1,2]) #Three classes for IPO, acquired, failure(?)
            cnn_output_size = self.cnn.hidden_layer_size // 2

        combination_input_size = self.d_model + cnn_output_size 

        #Final layer for processing
        self.hidden_layer = tf.keras.layers.Dense(
            combination_input_size, 
            activation = 'relu',
            name = 'combined_hidden_layer'
        )
        self.output_layer = tf.keras.layers.Dense(
            num_output_classes, 
            activation = 'softmax', 
            name = 'output')
        
    
        self.dropout_layer = tf.keras.layers.Dropout(
            self.dropout_rate
        )


    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        #TO DO: decide how best to store categorical inputs
        '''
        Docstring for call
        
        :param self: Description
        :param inputs: dictionary that holds the 'transformer_inputs' and 'cnn_inputs'
        :type inputs: dict[str, tf.Tensor]
        :param training: Description
        :type training: bool
        :return: Description
        :rtype: Any
        '''
        
        features_to_combine = []
        transformer_input = inputs['transformer_input']
        transformer_output = self.transformer(
            transformer_input,
            training=training
        )

        features_to_combine.append(transformer_output)
        if self.use_cnn and 'cnn_input' in inputs:
            cnn_input = inputs['cnn_input']
            if training is True:
                cnn_testing = False
            cnn_output = self.cnn.call(cnn_input, is_testing=cnn_testing)
            features_to_combine.append(cnn_output)

        combined = tf.concat(features_to_combine, axix=-1)
        x = self.hidden_layer(x)
        x = self.dropout_layer(x)
        output = self.output_layer(x)
        return output

        
    def get_config(self):
        '''Saves checkpoints for displaying values'''
        config = super().get_config()
        config.update({
            'categorical_vocab_sizes': self.vocab_size,
            'transformer_model': self.d_model,
            'use_cnn': self.use_cnn
        })

        
