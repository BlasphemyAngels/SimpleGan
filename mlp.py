# !/usr/bin/python3
# _*_coding: utf-8_*_

import tensorflow as tf

def mlp(input, output_dim):
    w1 = tf.get_variable("w0", [input.get_shape()[1], 6], initializer = tf.random_normal_initializer())
    b1 = tf.get_variable("b0", [6], initializer = tf.constant_initializer())
    w2 = tf.get_variable("w1", [6, 5], initializer = tf.random_normal_initializer())
    b2 = tf.get_variable("b1", [5], initializer = tf.constant_initializer())
    w3 = tf.get_variable("w2", [5, output_dim], initializer = tf.random_normal_initializer())
    b3 = tf.get_variable("b2", [output_dim], initializer = tf.constant_initializer())
    fc1 = tf.nn.tanh(tf.matmul(input, w1) + b1)
    fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)
    fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + b3)
    return fc3, [w1, b1, w2, b2, w3, b3]
