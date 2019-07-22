import tensorflow as tf


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


def get_logits(
    tokens,
    idx1,
    idx2,
    reuse,
    rnn_state_size=64,
    num_buckets=40000,
    num_hashes=3,
    embed_dim=12,
):
    """Compute logits Tensor using given feature Tensors.

    Args:
        tokens: [batch_size, num_tokens] Tensor of tf.string type.
        idx1: [batch_size, 2] int tensor representing person1_word_idx.
        idx2: [batch_size, 2] int tensor representing person2_word_idx.
        reuse: Whether to create new variables (False) or reuse existing variables (True).
        rnn_state_size: Integer. LSTM state size.
        num_buckets: Integer. Number of buckets to hash strings to integers.
        num_hashes: Integer. Number of distinct hash functions to use.
        embed_dim: Integer. Size of token embeddings.

    Returns:
        logits: [batch_size, 2] Tensor of output logits.
    """
    with tf.variable_scope("base", reuse=reuse):
        cell = tf.keras.layers.LSTMCell(rnn_state_size)
        rnn = tf.keras.layers.RNN(cell)

        # Set up one hot input for indexes
        num_words = tf.shape(tokens)[1]
        idx1_start = tf.one_hot(idx1[:, 0], num_words)
        idx1_end = tf.one_hot(idx1[:, 1], num_words)
        idx2_start = tf.one_hot(idx2[:, 0], num_words)
        idx2_end = tf.one_hot(idx2[:, 1], num_words)

        # Set up rnn inputs
        embeddings = [tf.stack([idx1_start, idx1_end, idx2_start, idx2_end], 2)]
        for i in range(num_hashes):
            ids = tf.strings.to_hash_bucket(str(i) + tokens, num_buckets)
            embedding_vars = tf.get_variable(
                shape=[num_buckets, embed_dim], name="embeddings_%d" % i
            )
            embeddings.append(tf.nn.embedding_lookup(embedding_vars, ids))
        embedded_input = tf.concat(embeddings, 2)
        output = rnn(embedded_input, mask=tf.strings.length(tokens))
        logits = tf.keras.layers.Dense(2)(output)
        return logits


def get_train_and_loss_op(tokens, idx1, idx2, label_probs, learn_rate=0.1):
    """Compute TensorFlow ops for training and getting cross entropy loss.

    Args:
        tokens: [batch_size, num_tokens] Tensor of tf.string type.
        idx1: [batch_size, 2] int tensor representing person1_word_idx.
        idx2: [batch_size, 2] int tensor representing person2_word_idx.
        label_probs: [batch_size, 2] tensorflow of label probabilities.
        learn_rate: Float. Learning rate.

    Returns:
        train_op: TensorFlow op for a single training step.
        mean_loss: Tensor representing average loss over a batch.
    """
    logits = get_logits(tokens, idx1, idx2, reuse=False)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(label_probs, logits)
    mean_loss = tf.reduce_mean(loss)
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdagradOptimizer(learn_rate)
    train_op = tf.group(global_step.assign_add(1), opt.minimize(loss))
    return train_op, mean_loss


def get_predictions_op(tokens, idx1, idx2):
    """
    Returns TensorFlow op for computing predicted labels for a batch.
    """
    logits = get_logits(tokens, idx1, idx2, reuse=True)
    return tf.argmax(logits, axis=1)
