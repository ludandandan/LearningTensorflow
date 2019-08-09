import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=False)

# Visualize decoder setting
# parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_input = 784#28*28

# tf grapy input (only picture)
X = tf.placeholder('float', [None, n_input])

# hidden layer settings
n_hidden_1 = 256 # 1st layer num feature
n_hidden_2 = 128
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases={
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input]))
}

# building the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                biases['encoder_b2']))
    return layer_2

# building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                biases['decoder_b2']))
    return layer_2

# construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# target are the input_data
y_true = X

# define loss and optimizer minimizer th squared error
cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# initializing the variable
init = tf.initialize_all_variables()

# launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # traing cycle
    for epoch in range(training_epochs):
        # loop over all batchs
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #max(x)=1, min(x)=0
            # run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # display logs per epoch step
        if epoch%display_step==0:
            print('Epoch:', '%04d' % (epoch+1),
                'cost=', '{:.9f}'.format(c))
    print('Optimization Finished!')
    
    # applying the encode and decode over testset
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]}
    )
    # compare original image with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10,2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28,28)))
    plt.show()

'''

# Visualize decoder setting
# parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_input = 784#28*28

# tf grapy input (only picture)
X = tf.placeholder('float', [None, n_input])

# hidden layer settings
n_hidden_1 = 256 # 1st layer num feature
n_hidden_2 = 2
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases={
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input]))
}

# building the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                biases['encoder_b2']))
    return layer_2

# building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                biases['decoder_b2']))
    return layer_2

# construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# target are the input_data
y_true = X

# define loss and optimizer minimizer th squared error
cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# initializing the variable
init = tf.initialize_all_variables()

# launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # traing cycle
    for epoch in range(training_epochs):
        # loop over all batchs
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #max(x)=1, min(x)=0
            # run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # display logs per epoch step
        if epoch%display_step==0:
            print('Epoch:', '%04d' % (epoch+1),
                'cost=', '{:.9f}'.format(c))
    print('Optimization Finished!')
    
    # applying the encode and decode over testset
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]}
    )
    # compare original image with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10,2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28,28)))
    plt.show()
'''
