import tensorflow as tf


def get_features_and_labels(features_df, labels, label_dtype):
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
    batch_size = 64

    def zip_feature_dicts(features):
        keys = sorted(features)
        val_lists = [features[k] for k in keys]
        keys = sorted(features)
        return lambda: (dict(zip(keys, vals)) for vals in zip(*val_lists))

    dataset = (
        tf.data.Dataset.from_generator(zip_feature_dicts(features), dtypes)
        .shuffle(123)
        .repeat(-1)
        .padded_batch(batch_size, {k: [None] for k in dtypes}, drop_remainder=True)
    )
    features_tensor_dict = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    idx1 = features_tensor_dict["person1_word_idx"]
    idx2 = features_tensor_dict["person2_word_idx"]
    tokens = features_tensor_dict["tokens"]
    label = features_tensor_dict["label"]
    return tokens, idx1, idx2, label


def get_logits(tokens, idx1, idx2, reuse):
    with tf.variable_scope("base", reuse=reuse):
        state_size = 64
        cell = tf.keras.layers.LSTMCell(state_size)
        rnn = tf.keras.layers.RNN(cell)

        # Set up one hot input for indexes
        num_words = tf.shape(tokens)[1]
        idx1_start = tf.one_hot(idx1[:, 0], num_words)
        idx1_end = tf.one_hot(idx1[:, 1], num_words)
        idx2_start = tf.one_hot(idx2[:, 0], num_words)
        idx2_end = tf.one_hot(idx2[:, 1], num_words)

        # Set up rnn inputs
        num_buckets = 40000
        num_hashes = 3
        embed_dim = 32
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


def get_train_and_loss_op(tokens, idx1, idx2, label_probs):
    logits = get_logits(tokens, idx1, idx2, reuse=False)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(label_probs, logits)
    mean_loss = tf.reduce_mean(loss)
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdagradOptimizer(0.1)
    train_op = tf.group(global_step.assign_add(1), opt.minimize(loss))
    return train_op, mean_loss


def get_predictions_op(tokens, idx1, idx2):
    logits = get_logits(tokens, idx1, idx2, reuse=True)
    return tf.argmax(logits, axis=1)
