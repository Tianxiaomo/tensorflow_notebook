# -*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
seed = 23455

rng = np.random.RandomState(seed)
X = rng.rand(32,2)

Y = [[int(x0 + x1 < 1)] for  (x0,x1) in X]
print(X)
print( Y)

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

W1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
W2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a = tf.matmul(x,W1)
y = tf.matmul(a,W2)

loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(W1))
    print(sess.run(W2))
    print('\n')

    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss,feed_dict = {x:X,y_:Y})
            print('已经训练了 %d 轮,loss : %g' % (i,total_loss))

    print(sess.run(W1))
    print(sess.run(W2))


