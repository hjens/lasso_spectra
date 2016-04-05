import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
import numpy as np

# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50

n_samples = 100

# Training Data
true_W = [[3.5]]
true_b = -8.5
n_features = len(true_W)
train_X = np.random.random((n_samples, n_features))*10.
train_Y = true_b + np.dot(train_X, true_W)

# tf Graph Input
X = tf.placeholder("float", [n_samples, n_features])
Y = tf.placeholder("float", [n_samples, 1])

# Create Model

# Set model weights
W = tf.Variable(tf.random_normal(shape=[n_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1]))

# Construct a linear model
#activation = tf.add(tf.mul(X, W), b)
activation = tf.add(tf.matmul(X, W), b)

# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        #Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y:train_Y})), \
                "W=", sess.run(W), "b=", sess.run(b)

        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})


    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

