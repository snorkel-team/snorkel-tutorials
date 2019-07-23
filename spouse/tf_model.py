import tensorflow as tf


def get_model(rnn_state_size=64, num_hashes=3, num_buckets=40000, embed_dim=124):
    """
    Return LSTM model for predicting label probabilities.

    Args:
        rnn_state_size: Integer. LSTM state size.
        num_hashes: Integer. Number of distinct hash functions to use.
        num_buckets: Integer. Number of buckets to hash strings to integers.
        embed_dim: Integer. Size of token embeddings.

    Returns:
        model: A compiled tf.keras.Model instance.
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
    model.compile(
        tf.train.AdagradOptimizer(0.1), "categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def get_features_and_labels(
    features_df, labels, label_dtype, batch_size=64, num_epochs=-1
):
    """
    Converts features dataframe and labels np.ndarray into tensorflow Tensors.

    Args:
        features_df: Dataframe where each row represents an example.
        labels: Either a [num_labels, 1] np.ndarray of integer labels or a
            [num_labels, 2] np.ndarray of float label probabilities.
        label_dtype: Tensorflow dtype for labels. e.g. tf.int64 or tf.float32.
        batch_size: Integer.
        num_epochs: Integer representing number of epochs over data. -1 means
            keep looping over data indefinitely.

    Returns:
        tokens: [batch_size, num_tokens] Tensor of tf.string type.
        idx1: [batch_size, 2] int tensor representing person1_word_idx.
        idx2: [batch_size, 2] int tensor representing person2_word_idx.
        label: [batch_size, <1 or 2>] tensor of type label_dtype.
    """
    features = {
        k: features_df[k].values
        for k in ["person1_word_idx", "person2_word_idx", "tokens"]
    }
    features["label"] = labels
    dtypes = {
        "person1_word_idx": tf.int64,
        "person2_word_idx": tf.int64,
        "tokens": tf.string,
        "label": label_dtype,
    }

    def zip_feature_dicts(features):
        keys = sorted(features)
        val_lists = [features[k] for k in keys]
        return lambda: (dict(zip(keys, vals)) for vals in zip(*val_lists))

    dataset = (
        tf.data.Dataset.from_generator(zip_feature_dicts(features), dtypes)
        .shuffle(123)
        .repeat(num_epochs)
        .padded_batch(batch_size, {k: [None] for k in dtypes}, drop_remainder=False)
    )
    features_tensor_dict = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    idx1 = features_tensor_dict["person1_word_idx"]
    idx2 = features_tensor_dict["person2_word_idx"]
    tokens = features_tensor_dict["tokens"]
    label = features_tensor_dict["label"]
    return tokens, idx1, idx2, label
