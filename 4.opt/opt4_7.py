#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE =30
seed = 2
STEPS = 40000

rdm = np.random.seed(seed)
X = rdm.randn(300,2)

Y_ = [int (x0*x0 + x1*x1 <2) for (x0,x1) in X]

Y_c = [['red' if y else 'blue'] for y in Y_]

X = np.vstack(X).reshape(-1,2)
Y_ = np.vstack(Y_).reshape(-1,1)

print(X.shape)
print(Y_.shape)
# print(Y_c)

plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.show()

def get_weight(shape,regularizer):
    W = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(W))
    return W

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape=shape))
    return b

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

W1 = get_weight([2,11],0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x,W1) + b1)

W2 = get_weight([11,1],0.001)
b2 = get_bias([1])
y = tf.matmul(y1,W2) + b2

loss_mse = tf.reduce_mean(tf.square(y - y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 无正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

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

# 有正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

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
