import tensorflow as tf
from numpy.random import RandomState
import numpy as np

w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

print(w1.dtype)
print(w1.shape)
print(w2.dtype)

x = tf.constant([[0.5, 0.3]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "save/save_net.ckpt")
    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(x))
    print(sess.run(y))
