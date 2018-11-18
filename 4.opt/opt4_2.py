# -*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
seed = 23455
LR = 0.001
STEPS = 20000

COST = 1
PROFIT = 9

rdm = np.random.seed(seed)
X = rdm.rand(32,2)
Y_ = [[x1 + x2 + (rdm.rand()/10 - 0.05)]for (x1,x2) in X]

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
W1 = tf.Variable(tf.random_normal([2,1],stddev = 1,seed=1))
y = tf.matmul(x,W1)

# 自定义loss
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(LR).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%32
        end = (i*BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i % 500 ==0:
            print('STEP:%d' % (i))
            print(sess.run(W1))
    print('train over')
    print(sess.run(W1))

"""
train over
[[1.020171 ]
 [1.0425103]]
"""