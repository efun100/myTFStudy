import tensorflow as tf
from numpy.random import RandomState
import numpy as np

w1 = tf.Variable(tf.random_normal([2, 4], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([4, 8], stddev = 1, seed = 1))
w3 = tf.Variable(tf.random_normal([8, 16], stddev = 1, seed = 1))
w4 = tf.Variable(tf.random_normal([16, 8], stddev = 1, seed = 1))
w5 = tf.Variable(tf.random_normal([8, 4], stddev = 1, seed = 1))
w6 = tf.Variable(tf.random_normal([4, 2], stddev = 1, seed = 1))
w7 = tf.Variable(tf.random_normal([2, 1], stddev = 1, seed = 1))

b1 = tf.Variable(tf.random_normal([1, 4], stddev=1, seed=1))
b2 = tf.Variable(tf.random_normal([1, 8], stddev=1, seed=1))
b3 = tf.Variable(tf.random_normal([1, 16], stddev=1, seed=1))
b4 = tf.Variable(tf.random_normal([1, 8], stddev=1, seed=1))
b5 = tf.Variable(tf.random_normal([1, 4], stddev=1, seed=1))
b6 = tf.Variable(tf.random_normal([1, 2], stddev=1, seed=1))
b7 = tf.Variable(tf.random_normal([1, 1], stddev=1, seed=1))

print(w1.dtype)
print(w1.shape)
print(w2.dtype)

x = tf.placeholder(tf.float32, shape = (None, 2), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'y-input')

a = tf.nn.relu(tf.matmul(x, w1) + b1)
b = tf.nn.relu(tf.matmul(a, w2) + b2)
c = tf.nn.relu(tf.matmul(b, w3) + b3)
d = tf.nn.relu(tf.matmul(c, w4) + b4)
e = tf.nn.relu(tf.matmul(d, w5) + b5)
f = tf.nn.relu(tf.matmul(e, w6) + b6)
y = tf.matmul(f, w7) + b7

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "save/save_net.ckpt")
    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(y, feed_dict={x: [[112.1, 7.1], [21.1, 337.1], [9.1, 19.0]]}))
