#coding:utf-8
# 设 loss = (w+1)^2 ,w = 5,反向传播求最有w，
import tensorflow as tf

#超参
BATCH_SIZE = 8
seed = 23455
LR = 0.1
STEPS = 40

W = tf.Variable(tf.constant(5,dtype = tf.float32))

loss = tf.square(W + 1)

train_step = tf.train.GradientDescentOptimizer(LR).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(STEPS):
        sess.run(train_step)
    w_val = sess.run(W)
    loss_val = sess.run(loss)
    print('Step:%d , W: %f ,Loss : %f' % (i,w_val,loss_val))

"""
Step:0 , W: 3.800000 ,Loss : 23.040001
Step:1 , W: 2.840000 ,Loss : 14.745600
Step:2 , W: 2.072000 ,Loss : 9.437184
Step:3 , W: 1.457600 ,Loss : 6.039798
Step:4 , W: 0.966080 ,Loss : 3.865470
Step:5 , W: 0.572864 ,Loss : 2.473901
Step:6 , W: 0.258291 ,Loss : 1.583297
Step:7 , W: 0.006633 ,Loss : 1.013310
Step:8 , W: -0.194694 ,Loss : 0.648518
Step:9 , W: -0.355755 ,Loss : 0.415052
Step:10 , W: -0.484604 ,Loss : 0.265633
Step:11 , W: -0.587683 ,Loss : 0.170005
Step:12 , W: -0.670147 ,Loss : 0.108803
Step:13 , W: -0.736117 ,Loss : 0.069634
Step:14 , W: -0.788894 ,Loss : 0.044566
Step:15 , W: -0.831115 ,Loss : 0.028522
Step:16 , W: -0.864892 ,Loss : 0.018254
Step:17 , W: -0.891914 ,Loss : 0.011683
Step:18 , W: -0.913531 ,Loss : 0.007477
Step:19 , W: -0.930825 ,Loss : 0.004785
Step:20 , W: -0.944660 ,Loss : 0.003063
Step:21 , W: -0.955728 ,Loss : 0.001960
Step:22 , W: -0.964582 ,Loss : 0.001254
Step:23 , W: -0.971666 ,Loss : 0.000803
Step:24 , W: -0.977333 ,Loss : 0.000514
Step:25 , W: -0.981866 ,Loss : 0.000329
Step:26 , W: -0.985493 ,Loss : 0.000210
Step:27 , W: -0.988394 ,Loss : 0.000135
Step:28 , W: -0.990716 ,Loss : 0.000086
Step:29 , W: -0.992572 ,Loss : 0.000055
Step:30 , W: -0.994058 ,Loss : 0.000035
Step:31 , W: -0.995246 ,Loss : 0.000023
Step:32 , W: -0.996197 ,Loss : 0.000014
Step:33 , W: -0.996958 ,Loss : 0.000009
Step:34 , W: -0.997566 ,Loss : 0.000006
Step:35 , W: -0.998053 ,Loss : 0.000004
Step:36 , W: -0.998442 ,Loss : 0.000002
Step:37 , W: -0.998754 ,Loss : 0.000002
Step:38 , W: -0.999003 ,Loss : 0.000001
Step:39 , W: -0.999202 ,Loss : 0.000001
"""