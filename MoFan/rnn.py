import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_inputs = 28 #28*28, input 1 row, n_inputs= cols
n_steps = 28 #time step, total 28 rows, n_steps=rows
n_hidden_units = 128 # neurons in hidden layer
n_classes = 10

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define Weights
weights={
    #(28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases={
    # (128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    #  (10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def RNN(X, weights, biases):
    # hidden layer for input to cell
    # X (128 batch, 28step(row), 28inputs(cols))
    # =>(128*28, 28inputs)
    X = tf.reshape(X, [-1, n_inputs])
    #X_in => (128batch*28steps, 128hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in => (128batch, 28steps, 128hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two part:(c_state, m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    
    # time_major: time step is in 0 dim?
    # outputs is a list ,
    # state is the final state
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    '''
    # hidden layer for output as the final results
    # states[1] = m_state
    results = tf.matmul(states[1], weight['out']) + biases['out']
    # results = m_state
    '''
    # or 
    # unpack to list [(batch, outputs)..]*steps
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2])) # state is the laset outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys
            })
        if step%20==0:
            print(sess.run(accuracy, feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }))

        step += 1
