# +
from typing import Tuple
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Embedding,
    Input,
    LSTM,
)


def get_feature_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get np arrays of upto max_length tokens and person idxs."""
    bet = df.between_tokens
    left = df.apply(lambda c: c.tokens[: c.person1_word_idx[0]][-4:-1], axis=1)
    right = df.person2_right_tokens

    def pad_or_truncate(l, max_length=40):
        return l[:max_length] + [""] * (max_length - len(l))

    left_tokens = np.array(list(map(pad_or_truncate, left)))
    bet_tokens = np.array(list(map(pad_or_truncate, bet)))
    right_tokens = np.array(list(map(pad_or_truncate, right)))
    return left_tokens, bet_tokens, right_tokens


def bilstm(
    tokens: tf.Tensor,
    rnn_state_size: int = 64,
    num_buckets: int = 40000,
    embed_dim: int = 36,
):
    ids = tf.strings.to_hash_bucket(tokens, num_buckets)
    embedded_input = Embedding(num_buckets, embed_dim)(ids)
    return Bidirectional(LSTM(rnn_state_size, activation=tf.nn.relu))(
        embedded_input, mask=tf.strings.length(tokens)
    )


def get_model(
    rnn_state_size: int = 64, num_buckets: int = 40000, embed_dim: int = 12
) -> tf.keras.Model:
    """
    Return LSTM model for predicting label probabilities.
    Args:
        rnn_state_size: LSTM state size.
        num_buckets: Number of buckets to hash strings to integers.
        embed_dim: Size of token embeddings.
    Returns:
        model: A compiled LSTM model.
    """
    left_ph = Input((None,), dtype="string")
    bet_ph = Input((None,), dtype="string")
    right_ph = Input((None,), dtype="string")
    left_embs = bilstm(left_ph, rnn_state_size, num_buckets, embed_dim)
    bet_embs = bilstm(bet_ph, rnn_state_size, num_buckets, embed_dim)
    right_embs = bilstm(right_ph, rnn_state_size, num_buckets, embed_dim)
    layer = Concatenate(1)([left_embs, bet_embs, right_embs])
    layer = Dense(64, activation=tf.nn.relu)(layer)
    layer = Dense(32, activation=tf.nn.relu)(layer)
    probabilities = Dense(2, activation=tf.nn.softmax)(layer)
    model = tf.keras.Model(inputs=[bet_ph, left_ph, right_ph], outputs=probabilities)
    model.compile(tf.train.AdagradOptimizer(0.1), "categorical_crossentropy")
    return model
