import gzip
import pickle

import tensorflow as tf
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
#    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


#f = gzip.open('mnist.pkl.gz', 'rb')
#train_set, valid_set, test_set = pickle.load(f)
#f.close()

with gzip.open('mnist.pkl.gz', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()

np.random.shuffle(list(train_set))
train_x, train_y = train_set

train_y = one_hot(train_y.astype(int), 10)

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import random

#plt.imshow(train_x[0].reshape((28, 28)))
#plt.show()
#print(train_y[0])
#plt.imshow(train_x[20].reshape((28, 28)))
#plt.show()
#print(train_y[20])
#for i in range(40):
#    plt.imshow(train_x[i].reshape((28, 28)), cmap=cm.Greys_r)
#    plt.show()  # Let's see a sample
#    print(train_y[i])

# ---------------- My neural Network model --------------

# Hyperparameters
h1_neurons = 20
h2_neurons = 10
BATCH_SIZE = 20
EPOCH = 20
learning_rate = 0.01
X = tf.placeholder(tf.float64, (None, 784))
Y_ = tf.placeholder(tf.float64, (None, 10))

W1 = tf.Variable(np.float64(np.random.rand(train_x.shape[1], h1_neurons)) * 0.1, dtype=tf.float64)
b1 = tf.Variable(np.float64(np.random.rand(h1_neurons)) * 0.1, dtype=tf.float64)

W2 = tf.Variable(np.float64(np.random.rand(h1_neurons, h2_neurons)) * 0.1, dtype=tf.float64)
b2 = tf.Variable(np.float64(np.random.rand(h2_neurons)) * 0.1, dtype=tf.float64)

h1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
Y = tf.nn.softmax(tf.matmul(h1, W2) + b2)

loss = tf.reduce_sum(tf.square(Y_ - Y))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    training_loss = list()
    tick = list()
    test_accuracy = list()
    currentLoss = None

    for _ in range(EPOCH):
        for i in range(len(train_x) // BATCH_SIZE):
            batch_X = train_x[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
            batch_Y = train_y[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
            out = sess.run(Y, feed_dict={X: batch_X, Y_: batch_Y})
            sess.run(train, feed_dict={X: batch_X, Y_: batch_Y})
            currentLoss = sess.run(loss, feed_dict={X: batch_X, Y_: batch_Y})

        print("[+] EPOCH : ", _ + 1, "Error: ", currentLoss)

        print(_)
        if _ % 5 == 0:
            training_loss.append(currentLoss)
            tick.append(_)

        result = sess.run(Y, feed_dict={X: batch_X})
        for b, r in zip(batch_Y, result):
            print(b, "-->", r)
        print("----------------------------------------------------------------------------------")

    plt.figure()
    plt.plot(training_loss, tick)
    plt.title("Epoch vs Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()