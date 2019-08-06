# +
import tensorflow as tf


def get_model(
    rnn_state_size: int = 64,
    num_hashes: int = 3,
    num_buckets: int = 40000,
    embed_dim: int = 12,
) -> tf.keras.Model:
    """
    Return LSTM model for predicting label probabilities.

    Args:
        rnn_state_size: LSTM state size.
        num_hashes: Number of distinct hash functions to use.
        num_buckets: Number of buckets to hash strings to integers.
        embed_dim: Size of token embeddings.

    Returns:
        model: A compiled LSTM model.
    """
    tokens_ph = tf.keras.layers.Input((None,), dtype="string")
    idx1_ph = tf.keras.layers.Input((2,), dtype="int64")
    idx2_ph = tf.keras.layers.Input((2,), dtype="int64")
    num_words = tf.shape(tokens_ph)[1]
    idx1_start = tf.one_hot(idx1_ph[:, 0], num_words)
    idx1_end = tf.one_hot(idx1_ph[:, 1], num_words)
    idx2_start = tf.one_hot(idx2_ph[:, 0], num_words)
    idx2_end = tf.one_hot(idx2_ph[:, 1], num_words)
    embeddings = [tf.stack([idx1_start, idx1_end, idx2_start, idx2_end], 2)]
    for i in range(num_hashes):
        ids = tf.strings.to_hash_bucket(str(i) + tokens_ph, num_buckets)
        embeddings.append(tf.keras.layers.Embedding(num_buckets, embed_dim)(ids))
    embedded_input = tf.concat(embeddings, 2)
    output = tf.keras.layers.LSTM(rnn_state_size, activation=tf.nn.relu)(
        embedded_input, mask=tf.strings.length(tokens_ph)
    )
    probabilities = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(output)
    model = tf.keras.Model(inputs=[tokens_ph, idx1_ph, idx2_ph], outputs=probabilities)
    model.compile(tf.train.AdagradOptimizer(0.1), "categorical_crossentropy")
    return model
