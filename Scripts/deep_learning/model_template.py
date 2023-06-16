#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tensorflow as tf


def make_model(vocab_size, embedding_dim, max_review_length, metrics, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=max_review_length, name="em_1"
        )
    )
    model.add(tf.keras.layers.LSTM(100, name="lstm_1", dropout=0.5))
    model.add(
        tf.keras.layers.Dense(
            1, activation="sigmoid", name="output", bias_initializer=output_bias
        )
    )

    print(model.summary())

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model


def make_model_multi_modal(
    vocab_size, embedding_dim, max_review_length, metrics, output_bias=None
):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=max_review_length, name="em_2"
        )
    )
    model.add(tf.keras.layers.LSTM(100, name="lstm_2", dropout=0.5))

    model.add(
        tf.keras.layers.Dense(
            1, activation="sigmoid", name="output", bias_initializer=output_bias
        )
    )

    model.add()

    print(model.summary())

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model


# In[ ]:
