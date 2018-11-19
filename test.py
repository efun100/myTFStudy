import tensorflow as tf
from numpy.random import RandomState
import numpy as np

w1 = tf.Variable(tf.random_normal([2, 4], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([4, 8], stddev = 1, seed = 1))
w3 = tf.Variable(tf.random_normal([8, 4], stddev = 1, seed = 1))
w4 = tf.Variable(tf.random_normal([4, 2], stddev = 1, seed = 1))
w5 = tf.Variable(tf.random_normal([2, 1], stddev = 1, seed = 1))

b1 = tf.Variable(tf.random_normal([1, 4], stddev=1, seed=1))
b2 = tf.Variable(tf.random_normal([1, 8], stddev=1, seed=1))
b3 = tf.Variable(tf.random_normal([1, 4], stddev=1, seed=1))
b4 = tf.Variable(tf.random_normal([1, 2], stddev=1, seed=1))
b5 = tf.Variable(tf.random_normal([1, 1], stddev=1, seed=1))

print(w1.dtype)
print(w1.shape)
print(w2.dtype)

x = tf.placeholder(tf.float32, shape = (None, 2), name = 'x-input')

a = tf.nn.relu(tf.matmul(x, w1) + b1)
b = tf.nn.relu(tf.matmul(a, w2) + b2)
c = tf.nn.relu(tf.matmul(b, w3) + b3)
d = tf.nn.relu(tf.matmul(c, w4) + b4)
y = tf.nn.relu(tf.matmul(d, w5) + b5)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "save/save_net.ckpt")
    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(y, feed_dict={x: [[11.1, 7.1], [21.1, 33.1], [9.1, 19.0]]}))
