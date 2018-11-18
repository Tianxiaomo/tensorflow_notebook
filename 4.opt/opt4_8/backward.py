#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateds
import forward

BATCH_SIZE =30
seed = 2
STEPS = 40000
LR_BASE = 0.001
LR_DECAY = 0.999
REGULARIZER = 0.01

def backward():
    x = tf.placeholder(tf.float32,shape=(None,2))
    y_ = tf.placeholder(tf.float32,shape=(None,1))
    X,Y_,Y_c = generateds.generateds()

    y = forward.forward(x,REGULARIZER)

    global_step = tf.Variable(0,trainable = False)

    lr = tf.train.exponential_decay(
        LR_BASE,
        global_step,
        300/BATCH_SIZE,
        LR_DECAY,
        staircase=True)

    loss_mse = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    # 有正则化
    train_step = tf.train.AdamOptimizer(lr).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i * BATCH_SIZE) %300
            end = start + BATCH_SIZE
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
            if i % 2000 == 0:
                loss_mse_v = sess.run(loss_mse,feed_dict={x:X,y_:Y_})
                print('Step:%d , loss：%f ' % (i,loss_mse_v))
        
        xx,yy = np.mgrid[-3:3:0.1,-3:3:0.1]
        grid = np.c_[xx.ravel(),yy.ravel()]
        probs = sess.run(y,feed_dict={x:grid})
        probs = probs.reshape(xx.shape)
    # print "W1:\n",sess.run(W1)
    # print "b1:\n",sess.run(b1)
    # print "W2:\n",sess.run(W2)
    # print "b2:\n",sess.run(b2)

    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show()

if __name__ == '__main__':
    backward()