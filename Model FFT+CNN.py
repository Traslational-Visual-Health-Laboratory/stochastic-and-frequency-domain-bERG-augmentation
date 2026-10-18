import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Concatenate,
    BatchNormalization, GRU, Dropout, Lambda, Multiply, Dense,
    LayerNormalization, MultiHeadAttention, TimeDistributed
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    x = LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    feed_forward = Dense(ff_dim, activation="relu")(x)
    feed_forward = Dropout(dropout)(feed_forward)
    feed_forward = Dense(inputs.shape[-1])(feed_forward)
    return LayerNormalization(epsilon=1e-6)(feed_forward + x)


def simple_conv_block(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = GlobalAveragePooling1D()(x)
    model = Model(inputs, x)
    return model


def multi_scale_conv_block(inputs, l2_lambda):
    conv_k5 = Conv1D(32, 5, padding='same', activation='relu',
                      kernel_regularizer=l2(l2_lambda))(inputs)

    conv_k25 = Conv1D(16, 25, padding='same', activation='relu',
                       kernel_regularizer=l2(l2_lambda))(inputs)

    conv_k100 = Conv1D(8, 100, padding='same', activation='relu',
                        kernel_regularizer=l2(l2_lambda))(inputs)

    x = Concatenate()([conv_k5, conv_k25, conv_k100])
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    return x


def cnn_branch_extractor(input_shape, l2_lambda=1e-4, dropout_rate=0.2):
    """CNN multi-escala + GRU + atención de canal (SE-like) para la señal ERG."""
    inputs = Input(shape=input_shape)
    x = multi_scale_conv_block(inputs, l2_lambda)
    x = GRU(32, return_sequences=True,
            kernel_regularizer=l2(l2_lambda),
            recurrent_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout_rate)(x)

    channel_attention = GlobalAveragePooling1D()(x)
    channel_attention = Dense(32, activation="elu", kernel_regularizer=l2(l2_lambda))(channel_attention)
    channel_attention = Dense(x.shape[-1], activation="sigmoid", kernel_regularizer=l2(l2_lambda))(channel_attention)
    channel_attention = Lambda(lambda a: tf.expand_dims(a, axis=1))(channel_attention)

    x = Multiply()([x, channel_attention])
    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)

    model = Model(inputs, x)
    return model


def build_fft_block(input_shape, sampling_freq=200, max_freq=40, name="fft_layer"):
    return FFTLayer(
        window_size=input_shape[0],
        sampling_freq=sampling_freq,
        max_freq=max_freq,
        name=name
    )


class FFTLayer(tf.keras.layers.Layer):
    def __init__(self,
                 window_size,
                 sampling_freq=200,
                 max_freq=40,
                 name="fft_attention_layer",
                 value_channels=8,
                 attn_channels=8,
                 delta_init_std=1e-3,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.window_size = int(window_size)
        self.sampling_freq = float(sampling_freq)
        self.max_freq = float(max_freq)
        self._name = name

        self.max_index = int(np.floor(self.max_freq * self.window_size / self.sampling_freq)) + 1
        if self.max_index < 2:
            self.max_index = 2

        self.attn_channels = int(attn_channels)
        self.value_channels = int(value_channels)
        self.delta_init_std = float(delta_init_std)

        self.conv_query = Conv1D(self.attn_channels, kernel_size=1, padding="same", name=f"{self._name}_query")
        self.conv_key = Conv1D(self.attn_channels, kernel_size=1, padding="same", name=f"{self._name}_key")
        self.conv_value = Conv1D(self.value_channels, kernel_size=1, padding="same", name=f"{self._name}_value")

        self.conv_residual = Conv1D(self.value_channels, kernel_size=1, padding="same", name=f"{self._name}_res_proj")

        self.ln_pre = LayerNormalization(name=f"{self._name}_ln_pre")
        self.ln_post = LayerNormalization(name=f"{self._name}_ln_post")

        self.band_ranges = [(0.3, 4), (4, 8), (8, 12), (13, 30), (30, max_freq)]

    def build(self, input_shape):
        self.delta_f = self.add_weight(
            shape=(self.max_index,),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.delta_init_std),
            trainable=True,
            name=f"{self._name}_delta_f"
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = tf.squeeze(inputs, -1) if inputs.shape.rank == 3 and inputs.shape[-1] == 1 else inputs
        x = tf.cast(x, tf.float32)

        fft_complex = tf.signal.rfft(x)
        fft_magnitude = tf.abs(fft_complex)
        fft_magnitude = fft_magnitude[:, :self.max_index]
        fft_real = tf.expand_dims(fft_magnitude, -1)

        n = tf.cast(tf.range(self.window_size), tf.float32)
        k = tf.cast(tf.range(self.max_index), tf.float32)
        base_freq = k * (self.sampling_freq / self.window_size)

        learned_freq = base_freq + self.delta_f

        cos_matrix = tf.cos(2.0 * np.pi * tf.reshape(learned_freq, (-1, 1)) * tf.reshape(n, (1, -1)))
        sin_matrix = tf.sin(2.0 * np.pi * tf.reshape(learned_freq, (-1, 1)) * tf.reshape(n, (1, -1)))
        cos_matrix = tf.expand_dims(cos_matrix, 0)
        sin_matrix = tf.expand_dims(sin_matrix, 0)

        x_expanded = tf.expand_dims(x, 1)
        real_part = tf.reduce_sum(x_expanded * cos_matrix, axis=-1)
        imag_part = tf.reduce_sum(x_expanded * sin_matrix, axis=-1)
        learned_magnitude = tf.sqrt(real_part ** 2 + imag_part ** 2 + 1e-8)

        fft_learned = tf.expand_dims(learned_magnitude, -1)

        fused_spectrum = tf.concat([fft_real, fft_learned], axis=-1)

        x_norm = self.ln_pre(fused_spectrum)
        query = self.conv_query(x_norm)
        key = self.conv_key(x_norm)
        value = self.conv_value(x_norm)

        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(self.attn_channels, tf.float32) + 1e-9)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        weighted_value = tf.matmul(attention_weights, value)

        residual = self.conv_residual(x_norm)
        attended_output = self.ln_post(residual + weighted_value)

        magnitude = tf.reduce_mean(tf.abs(attended_output), axis=-1)

        avg_pool = tf.reduce_mean(magnitude, axis=1, keepdims=True)
        std_pool = tf.math.reduce_std(magnitude, axis=1, keepdims=True)
        energy_pool = tf.reduce_sum(magnitude ** 2, axis=1, keepdims=True)

        top_k_values, top_k_indices = tf.math.top_k(magnitude, k=5, sorted=True)
        top_k_freqs = tf.gather(learned_freq, top_k_indices)

        band_means = []
        expanded_freqs = tf.reshape(learned_freq, (1, -1))

        for low_freq, high_freq in self.band_ranges:
            band_mask = tf.logical_and(expanded_freqs >= low_freq, expanded_freqs < high_freq)
            band_mask_float = tf.cast(band_mask, tf.float32)
            band_sum = tf.reduce_sum(magnitude * band_mask_float, axis=1, keepdims=True)
            band_count = tf.reduce_sum(band_mask_float)
            band_means.append(band_sum / tf.maximum(band_count, 1.0))

        band_means = tf.concat(band_means, axis=-1)
        output = tf.concat([
            avg_pool,
            std_pool,
            energy_pool,
            band_means,
            tf.cast(top_k_values, tf.float32),
            tf.cast(top_k_freqs, tf.float32)
        ], axis=-1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 18)

    def get_config(self):
        config = super().get_config()
        config.update({
            "window_size": int(self.window_size),
            "sampling_freq": float(self.sampling_freq),
            "max_freq": float(self.max_freq),
            "attn_channels": self.attn_channels,
            "value_channels": self.value_channels,
            "delta_init_std": self.delta_init_std,
            "name": self._name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AttentionPooling1D(tf.keras.layers.Layer):
    """
    Attention-based pooling over the temporal dimension (num_windows).
    Learns an importance score per time step and computes a weighted sum,
    instead of a fixed average or max.
    """
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.score_dense = Dense(self.units, activation="relu")
        self.score_out = Dense(1)
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (batch, num_windows, features)
        scores = self.score_dense(x)               # (batch, num_windows, units)
        scores = self.score_out(scores)             # (batch, num_windows, 1)
        weights = tf.nn.softmax(scores, axis=1)     # normalize over num_windows
        pooled = tf.reduce_sum(x * weights, axis=1)  # (batch, features)
        return pooled, weights

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def build_cnn_fft_model(input_shape, num_classes, window_size, dropout=0.2):
    inputs_signal = Input(shape=(None, window_size, input_shape[-1]))

    # CNN branch
    x_cnn = TimeDistributed(cnn_branch_extractor((window_size, input_shape[-1])))(inputs_signal)
    x_cnn = TimeDistributed(BatchNormalization())(x_cnn)

    # FFT branch
    x_fft = TimeDistributed(build_fft_block((window_size, input_shape[-1])))(inputs_signal)
    x_fft = LayerNormalization(epsilon=1e-6)(x_fft)

    # Individual transformer encoders
    x_cnn = transformer_encoder_block(x_cnn, head_size=64, num_heads=4, ff_dim=128, dropout=dropout)
    x_fft = transformer_encoder_block(x_fft, head_size=64, num_heads=4, ff_dim=128, dropout=dropout)

    # Cross-attention CNN <-> FFT
    fft_to_cnn = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(query=x_fft, key=x_cnn, value=x_cnn)
    cnn_to_fft = MultiHeadAttention(num_heads=4, key_dim=64, dropout=dropout)(query=x_cnn, key=x_fft, value=x_fft)

    x_fft = LayerNormalization()(x_fft + fft_to_cnn)
    x_cnn = LayerNormalization()(x_cnn + cnn_to_fft)

    # Fusion of both branches
    x_signal = Concatenate()([x_fft, x_cnn])  # (batch, num_windows, features)

    # Attention pooling: keeps the most informative window(s)
    x, attention_weights = AttentionPooling1D(units=64)(x_signal)

    x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(num_classes, activation="sigmoid" if num_classes == 1 else "softmax")(x)

    model = Model(inputs=inputs_signal, outputs=outputs, name="cnn_fft_model")
    return model