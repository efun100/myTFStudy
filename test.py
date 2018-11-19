import tensorflow as tf
from numpy.random import RandomState
import numpy as np

w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))
b = tf.Variable(tf.random_normal([1, 1], stddev=1, seed=1))

print(w1.dtype)
print(w1.shape)
print(w2.dtype)

x = tf.placeholder(tf.float32, shape = (None, 2), name = 'x-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2) + b

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "save/save_net.ckpt")
    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(y, feed_dict={x: [[0.417, 0.72], [0.114, 0.302], [0.147, 0.09]]}))
