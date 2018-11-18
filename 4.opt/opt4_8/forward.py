#coding:utf-8
import tensorflow as tf
def get_weight(shape,regularizer):
    W = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(W))
    return W

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape=shape))
    return b

def forward(x,regularizer):
    W1 = get_weight([2,11],regularizer)
    b1 = get_bias([11])
    y1 = tf.nn.relu(tf.matmul(x,W1) + b1)

    W2 = get_weight([11,1],regularizer)
    b2 = get_bias([1])
    y = tf.matmul(y1,W2) + b2

    return y