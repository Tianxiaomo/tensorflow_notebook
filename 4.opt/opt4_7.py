#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE =30
seed = 2

rdm = np.random.RandomState(seed)
X = rdm.randn(300,2)

Y_ = [int (x0*x0 + x1*x1 <2) for (x0,x1) in X]

Y_c = [['red' if y else 'blue'] for y in Y_]

X = np.vstack(X).reshape(-1,2)
Y_ = np.vstack(Y_).reshape(-1,1)

print(X)
print(Y_)
print(Y_c)

plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.show()

