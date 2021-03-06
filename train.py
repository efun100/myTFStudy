import tensorflow as tf
from numpy.random import RandomState
import numpy as np

batch_size = 16

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

cross_entropy = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

rdm = RandomState(1)

dataset_size = 1024

X = rdm.uniform(1, 300, (dataset_size, 2))
print(X)

Y = [[x1 - x2] for (x1, x2) in X]
print(Y)

saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 100000

    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_:Y[start:end]})

        if i % 1000 == 0:
            total_cross_entropy=sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print("after %d training step, cross_entropy on all data is %g" % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))

    save_path = saver.save(sess, "save/save_net.ckpt")
    print("Save to path: ", save_path)

