# startup_transformer.py
import tensorflow as tf

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Self-attention
        attn_output = self.mha(x, x, x)  # (batch, seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)

        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)

        return out2


class StartupTransformer(tf.keras.Model):
    """
    Transformer-based classifier for startup success.
    Inputs:
      - categorical_inputs: (batch, num_cat_features) integer IDs
      - numeric_inputs: (batch, num_numeric_features) floats
    Output:
      - logits: (batch, num_classes)
    """
    def __init__(
        self,
        num_cat_features,
        num_numeric_features,
        vocab_sizes,
        num_classes,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert len(vocab_sizes) == num_cat_features, "vocab_sizes length must match num_cat_features"

        self.num_cat_features = num_cat_features
        self.num_numeric_features = num_numeric_features
        self.num_classes = num_classes
        self.d_model = d_model

        # One embedding layer per categorical feature
        self.cat_embeddings = []
        for i, vocab_size in enumerate(vocab_sizes):
            emb = tf.keras.layers.Embedding(
                input_dim=vocab_size + 1,   # +1 just in case
                output_dim=d_model,
                name=f"cat_emb_{i}",
            )
            self.cat_embeddings.append(emb)

        # Positional embeddings for each feature position (0..num_cat_features-1)
        self.pos_embedding = tf.keras.layers.Embedding(
            input_dim=num_cat_features,
            output_dim=d_model,
            name="pos_embedding",
        )

        # Transformer encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                name=f"encoder_layer_{i}",
            )
            for i in range(num_layers)
        ]

        # Projection for numeric data
        if num_numeric_features > 0:
            self.numeric_mlp = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dropout(dropout_rate),
                    tf.keras.layers.Dense(64, activation="relu"),
                ],
                name="numeric_mlp",
            )
        else:
            self.numeric_mlp = None

        # Final classifier
        combined_dim = d_model + (64 if self.numeric_mlp is not None else 0)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_classes, name="output_logits")

    def call(self, inputs, training=False):
        cat_inputs = inputs["categorical_inputs"]  # (batch, num_cat_features)
        num_inputs = inputs.get("numeric_inputs", None)

        batch_size = tf.shape(cat_inputs)[0]
        seq_len = tf.shape(cat_inputs)[1]  # == num_cat_features

        # Embed each categorical feature separately and stack
        emb_list = []
        for i, emb in enumerate(self.cat_embeddings):
            token_ids = cat_inputs[:, i]              # (batch,)
            e = emb(token_ids)                        # (batch, d_model)
            e = tf.expand_dims(e, axis=1)            # (batch, 1, d_model)
            emb_list.append(e)

        x = tf.concat(emb_list, axis=1)              # (batch, seq_len, d_model)

        # Add positional embeddings (feature index)
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = self.pos_embedding(positions)      # (seq_len, d_model)
        pos_emb = tf.expand_dims(pos_emb, axis=0)    # (1, seq_len, d_model)
        x = x + pos_emb

        # Transformer encoder stack
        for layer in self.encoder_layers:
            x = layer(x, training=training)          # (batch, seq_len, d_model)

        # Pool over sequence (mean pooling)
        x_pooled = tf.reduce_mean(x, axis=1)         # (batch, d_model)

        # Numeric branch
        if self.numeric_mlp is not None and num_inputs is not None:
            num_repr = self.numeric_mlp(num_inputs, training=training)  # (batch, 64)
            combined = tf.concat([x_pooled, num_repr], axis=-1)         # (batch, d_model + 64)
        else:
            combined = x_pooled

        combined = self.dropout(combined, training=training)
        logits = self.classifier(combined)           # (batch, num_classes)
        return logits
