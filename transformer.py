
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="transformer")
class AttentionMatrix(keras.layers.Layer):
    """Compute attention matrix"""

    def __init__(self, use_causal_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.use_causal_mask = use_causal_mask

    def call(self, inputs):
        """
        Compute attention weights from K and Q matrices.

        Args:
            inputs: [K, Q] where K and Q are [batch_size, seq_length, embed_size]

        Returns:
            attention_weights: [batch_size, seq_length, seq_length]
        """
        K, Q = inputs

        # 1. Ensure consistent dtypes (cast to tf.float32)
        K = tf.cast(K, tf.float32)
        Q = tf.cast(Q, tf.float32)
        head_size = tf.cast(tf.shape(K)[-1], tf.float32)
        
        # TODO: Compute scaled dot-product attention scores and normalize
        attention_weights = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(head_size)

        # TODO: If use_causal_mask is True, apply causal mask to prevent attending to future tokens
        if self.use_causal_mask is True:
            seq_length = tf.shape(attention_weights)[-1]
            mask_skeleton = tf.ones((seq_length, seq_length))
            mask = tf.linalg.band_part(mask_skeleton, -1, 0)
            attention_weights = tf.where(
                tf.equal(mask, 0), tf.constant(float('-inf')),
                attention_weights
            )
            
        # TODO: Apply softmax to get attention weights
        normalized = tf.nn.softmax(attention_weights, axis = -1)
        return normalized 

    def get_config(self):
        config = super().get_config()
        return config

@keras.saving.register_keras_serializable(package="transformer")
class AttentionHead(keras.layers.Layer):
    """Single attention head"""

    def __init__(self, input_size, output_size, use_causal_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.use_causal_mask = use_causal_mask

        self.use_causal_mask = use_causal_mask
        self.K_matrix = self.add_weight(shape = (input_size, output_size), initializer = "glorot_uniform", trainable = True, name = "k_matrix")
        self.Q_matrix = self.add_weight(shape = (input_size, output_size), initializer = "glorot_uniform", trainable = True, name = "q_matrix")
        self.V_matrix = self.add_weight(shape = (input_size, output_size), initializer = "glorot_uniform", trainable = True, name = "v_matrix")
        # TODO: Initialize linear projections for K, Q, V
        # TODO: Initialize attention matrix computation (pass in use_causal_mask)
        self.attention_matrix = AttentionMatrix(use_causal_mask=use_causal_mask)

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Apply single attention head.

        Args:
            inputs_for_keys: [batch_size, seq_length, input_size]
            inputs_for_values: [batch_size, seq_length, input_size]
            inputs_for_queries: [batch_size, seq_length, input_size]

        Returns:
            output: [batch_size, seq_length, output_size]
        """
        # 1. Ensure consistent dtypes
        inputs_for_keys = tf.cast(inputs_for_keys, tf.float32)
        inputs_for_values = tf.cast(inputs_for_values, tf.float32)
        inputs_for_queries = tf.cast(inputs_for_queries, tf.float32)

        K = tf.matmul(inputs_for_keys, self.K_matrix)
        Q = tf.matmul(inputs_for_queries, self.Q_matrix)
        V = tf.matmul(inputs_for_values, self.V_matrix)

        attention_weights = self.attention_matrix([K,Q])
        output = tf.matmul(attention_weights, V)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "output_size": self.output_size,
            "use_causal_mask": self.use_causal_mask
        })
        return config

@keras.saving.register_keras_serializable(package="transformer")
class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention mechanism"""

    def __init__(self, embed_size, num_heads=8, use_causal_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.use_causal_mask = use_causal_mask

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.attention_heads = []
        for n in range(num_heads):
            self.attention_heads.append(AttentionHead(self.embed_size, self.head_size, use_causal_mask=use_causal_mask))
        # TODO: Initialize output projection (embed_size)
        self.output_projection = tf.keras.layers.Dense(embed_size, use_bias=False) #Dense layer allows for the parameters within the output projection to be trainable 

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Apply multi-head attention.

        Args:
            inputs_for_keys: [batch_size, seq_length, embed_size]
            inputs_for_values: [batch_size, seq_length, embed_size]
            inputs_for_queries: [batch_size, seq_length, embed_size]

        Returns:
            output: [batch_size, seq_length, embed_size]
        """
        # 1. Ensure consistent dtypes
        inputs_for_keys = tf.cast(inputs_for_keys, tf.float32)
        inputs_for_values = tf.cast(inputs_for_values, tf.float32)
        inputs_for_queries = tf.cast(inputs_for_queries, tf.float32)

        outputs = []
        for head in self.attention_heads:
            output = head(inputs_for_keys, inputs_for_values, inputs_for_queries)
            outputs.append(output)

        return self.output_projection(tf.concat(outputs, axis = -1))

        # TODO: Apply each attention head
        # TODO: Concatenate head outputs
        # TODO: Apply output projection

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "num_heads": self.num_heads,
            "use_causal_mask": self.use_causal_mask
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding for transformer inputs.
    Uses sinusoidal position encodings as described in "Attention Is All You Need".
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Create positional encoding matrix
        pe = self.get_positional_encoding(max_seq_length, d_model)
        self.positional_encoding = tf.Variable(
            initial_value=pe, trainable=False, name='positional_encoding'
        )


    def get_positional_encoding(self, seq_length: int, d_model: int) -> tf.Tensor:
        """Generate sinusoidal positional encodings.
        
        Here is a link to the tensorflow implementation we are following:
        https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
        """
        
        # This is a position matrix where each row is position 0,1,2,...seq_length-1
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]  # [seq_length, 1]

        # This is the division term for the angles (computed from the Attention is All You Need paper)
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))  # [d_model // 2]
        
        # This calculates the sine and cosine terms for even and odd indices respectively
        pe_sin = tf.sin(position * div_term)  # [seq_length, d_model // 2]
        pe_cos = tf.cos(position * div_term)  # [seq_length, d_model // 2]

        # Now we stack and reshape to get the final positional encoding matrix
        pe = tf.stack([pe_sin, pe_cos], axis=2)  # [seq_length, d_model // 2, 2]
        pe = tf.reshape(pe, [seq_length, d_model])  # [seq_length, d_model]

        # NOTE: We need to add a batch dimension for broadcasting later it to inputs        
        return tf.expand_dims(pe, axis=0)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            inputs with positional encodings added
        """
        seq_length = tf.shape(inputs)[1]
        #d_model = tf.cast(tf.shape(inputs)[2], tf.float32)
        #pe = self.get_positional_encoding(tf.cast(seq_length, tf.float32), d_model)
        pe = self.positional_encoding[:,:seq_length,:]
        # TODO: Extract appropriate slice of positional encodings
        # HINT: self.positional_encoding has shape [1, max_seq_length, d_model]
        return tf.cast(pe + inputs, tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_seq_length": self.max_seq_length
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class LanguageTransformerBlock(keras.layers.Layer):
    """Single transformer block optimized for language modeling (no cross-attention)"""

    def __init__(self, embed_size, num_heads=8, ff_hidden_size=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_hidden_size = ff_hidden_size or 4 * embed_size
        self.dropout_rate = dropout_rate
        self.use_causal_mask = True  # Always use causal mask for language modeling

        # TODO: Initialize self-attention
        self.self_attention = MultiHeadAttention(self.embed_size, self.num_heads, self.use_causal_mask)
        # TODO: Initialize feed-forward network (2 layers)
        # First layer: embed_size -> ff_hidden_size with activation
        self.hidden_layer = tf.keras.layers.Dense(self.ff_hidden_size)
        # Second layer: ff_hidden_size -> embed_size
        self.hidden_layer_2 = tf.keras.layers.Dense(self.embed_size)
        # TODO: Initialize layer normalization layers
        self.normalization_layer = tf.keras.layers.LayerNormalization(axis = -1)
        self.normalization_layer_2 = tf.keras.layers.LayerNormalization(axis = -1)

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout_layer_2 = tf.keras.layers.Dropout(self.dropout_rate)
        # TODO: Initialize dropout layers

    def call(self, inputs, training=None):
        """
        Apply transformer block with residual connections and layer normalization.

        Args:
            inputs: [batch_size, seq_length, embed_size]
            training: Whether in training mode

        Returns:
            output: [batch_size, seq_length, embed_size]
        """
        # 1. Ensure consistent dtype, inputs = input
        inputs = tf.cast(inputs, tf.float32)
        self_attention = self.self_attention(inputs, inputs, inputs)
        dropout = self.dropout_layer(self_attention)
        residual_conn = self.normalization_layer(dropout + inputs)

        #feedforward_network (treats two layers as one)
        forward_1 = self.hidden_layer(residual_conn)
        forward_1 = tf.nn.relu(forward_1)
        forward_2 = self.hidden_layer_2(forward_1)
        dropout_2 = self.dropout_layer_2(forward_2)
        residual_conn_2 = self.normalization_layer_2(dropout_2 + residual_conn)


        return tf.cast(residual_conn_2, tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "num_heads": self.num_heads,
            "ff_hidden_size": self.ff_hidden_size,
            "dropout_rate": self.dropout_rate
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class TransformerLanguageModel(keras.Model):
    """
    Complete Transformer Language Model
    """

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=None,
                 max_seq_length=512, dropout_rate=0.1, pad_token_id=0, **kwargs):
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.pad_token_id = pad_token_id

        # TODO: Initialize token embeddings
        self.token_embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model, mask_zero=True, name = "embeddings_layer")
        # TODO: Create positional encodings (d_model, max_seq_length)
        self.positional_encodings = PositionalEncoding(self.d_model, self.max_seq_length)
        # TODO: Initialize embedding dropout
        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout_rate, name = "embedding_dropout")
        # TODO: Create transformer blocks (n_layers of LanguageTransformerBlock)
        self.transformer_blocks = [LanguageTransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout_rate, name = f"transformer_block_{i}") for i in range(self.n_layers)]
        # TODO: Initialize final layer normalization
        self.final_normalization = tf.keras.layers.LayerNormalization(axis=-1)
        # TODO: Initialize transformer dropout
        self.final_transformer_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        # TODO: Initialize output projection to vocabulary
        self.output_projection = tf.keras.layers.Dense(self.vocab_size, use_bias=False)

    def call(self, inputs, training=None):
        """
        Forward pass through the language model.

        Args:
            inputs: Token indices [batch_size, seq_length]
            training: Whether in training mode

        Returns:
            Logits over vocabulary [batch_size, seq_length, vocab_size]
        """
        # 1. Get token embeddings and scale by sqrt(d_model)
        embeddings = self.token_embeddings(inputs)  # [batch_size, seq_length, d_model]
        embeddings = embeddings * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Positional encodings have to be sliced to accomodate the possibility that the string may not be equal to the maximum sequence
        seq_length = tf.shape(inputs)[1]
        positional_encoding = self.positional_encodings(embeddings)
        # TODO: Apply dropout to embeddings
        x = self.embedding_dropout(positional_encoding)
        # TODO: Pass embeddings through transformer blocks using a loop
        for block in self.transformer_blocks:
            x = block(x, training=training)
        # TODO: Apply final layer normalization
        x = self.final_normalization(x)
        # TODO: Project to vocabulary and return logits
        x = self.final_transformer_dropout(x, training=training)
        logits = self.output_projection(x)
        return logits

    def get_config(self):
        """Get model configuration for saving."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'max_seq_length': self.max_seq_length,
            'dropout_rate': self.dropout_rate,
            'pad_token_id': self.pad_token_id
        }

def create_language_model(vocab_size: int, **kwargs) -> TransformerLanguageModel:
    """
    Factory function to create a language model with sensible defaults.

    Args:
        vocab_size: Size of the vocabulary
        **kwargs: Additional model parameters

    Returns:
        Initialized TransformerLanguageModel
    """
    default_config = {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_length': 256,
        'dropout_rate': 0.1,
        'pad_token_id': 0
    }

    # Update with provided kwargs
    config = {**default_config, **kwargs}
    config['vocab_size'] = vocab_size

    return TransformerLanguageModel(**config)