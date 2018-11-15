#coding:utf-8
# 设 loss = (w+1)^2 ,w = 5,反向传播求最有w，
import tensorflow as tf

#超参
BATCH_SIZE = 8
seed = 23455
lr_base = 0.1           # 基础学习率
lr_rate_decay = 0.99    #衰减率
lr_rate_step = 1        #多少轮衰减一次
STEPS = 40

global_step = tf.Variable(0,trainable = False)

lr = tf.train.exponential_decay(lr_base,global_step,lr_rate_step,lr_rate_decay,staircase = True)

W = tf.Variable(tf.constant(5,dtype = tf.float32))

loss = tf.square(W + 1)

train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(STEPS):
        sess.run(train_step)
        global_step_val = sess.run(global_step)
        w_val = sess.run(W)
        loss_val = sess.run(loss)
        print('Step:%d ,Global_step: %f , W: %f ,Loss : %f' % (i,global_step_val,w_val,loss_val))

"""
Step:0 ,Global_step: 1.000000 , W: 3.800000 ,Loss : 23.040001
Step:1 ,Global_step: 2.000000 , W: 2.849600 ,Loss : 14.819419
Step:2 ,Global_step: 3.000000 , W: 2.095001 ,Loss : 9.579033
Step:3 ,Global_step: 4.000000 , W: 1.494386 ,Loss : 6.221961
Step:4 ,Global_step: 5.000000 , W: 1.015167 ,Loss : 4.060896
Step:5 ,Global_step: 6.000000 , W: 0.631886 ,Loss : 2.663051
Step:6 ,Global_step: 7.000000 , W: 0.324608 ,Loss : 1.754587
Step:7 ,Global_step: 8.000000 , W: 0.077684 ,Loss : 1.161403
Step:8 ,Global_step: 9.000000 , W: -0.121202 ,Loss : 0.772287
Step:9 ,Global_step: 10.000000 , W: -0.281761 ,Loss : 0.515867
Step:10 ,Global_step: 11.000000 , W: -0.411674 ,Loss : 0.346128
Step:11 ,Global_step: 12.000000 , W: -0.517024 ,Loss : 0.233266
Step:12 ,Global_step: 13.000000 , W: -0.602644 ,Loss : 0.157891
Step:13 ,Global_step: 14.000000 , W: -0.672382 ,Loss : 0.107334
Step:14 ,Global_step: 15.000000 , W: -0.729305 ,Loss : 0.073276
Step:15 ,Global_step: 16.000000 , W: -0.775868 ,Loss : 0.050235
Step:16 ,Global_step: 17.000000 , W: -0.814036 ,Loss : 0.034583
Step:17 ,Global_step: 18.000000 , W: -0.845387 ,Loss : 0.023905
Step:18 ,Global_step: 19.000000 , W: -0.871193 ,Loss : 0.016591
Step:19 ,Global_step: 20.000000 , W: -0.892476 ,Loss : 0.011561
Step:20 ,Global_step: 21.000000 , W: -0.910065 ,Loss : 0.008088
Step:21 ,Global_step: 22.000000 , W: -0.924629 ,Loss : 0.005681
Step:22 ,Global_step: 23.000000 , W: -0.936713 ,Loss : 0.004005
Step:23 ,Global_step: 24.000000 , W: -0.946758 ,Loss : 0.002835
Step:24 ,Global_step: 25.000000 , W: -0.955125 ,Loss : 0.002014
Step:25 ,Global_step: 26.000000 , W: -0.962106 ,Loss : 0.001436
Step:26 ,Global_step: 27.000000 , W: -0.967942 ,Loss : 0.001028
Step:27 ,Global_step: 28.000000 , W: -0.972830 ,Loss : 0.000738
Step:28 ,Global_step: 29.000000 , W: -0.976931 ,Loss : 0.000532
Step:29 ,Global_step: 30.000000 , W: -0.980378 ,Loss : 0.000385
Step:30 ,Global_step: 31.000000 , W: -0.983281 ,Loss : 0.000280
Step:31 ,Global_step: 32.000000 , W: -0.985730 ,Loss : 0.000204
Step:32 ,Global_step: 33.000000 , W: -0.987799 ,Loss : 0.000149
Step:33 ,Global_step: 34.000000 , W: -0.989550 ,Loss : 0.000109
Step:34 ,Global_step: 35.000000 , W: -0.991035 ,Loss : 0.000080
Step:35 ,Global_step: 36.000000 , W: -0.992297 ,Loss : 0.000059
Step:36 ,Global_step: 37.000000 , W: -0.993369 ,Loss : 0.000044
Step:37 ,Global_step: 38.000000 , W: -0.994284 ,Loss : 0.000033
Step:38 ,Global_step: 39.000000 , W: -0.995064 ,Loss : 0.000024
Step:39 ,Global_step: 40.000000 , W: -0.995731 ,Loss : 0.000018
"""