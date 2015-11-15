import json
import input_data
import tensorflow as tf

def main():
    # Load data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Start TensorFLow session
    sess = tf.InteractiveSession()

    # Define input / output
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    # Hidden layer 1
    W1 = tf.Variable(tf.zeros([784, 25]))
    b1 = tf.Variable(tf.zeros([25]))
    h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    # Apply dropout to prevent overfitting
    keep_prob = tf.placeholder("float")
    h_drop = tf.nn.dropout(h1, keep_prob)

    # Readout layer
    W3 = tf.Variable(tf.zeros([25, 10]))
    b3 = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(h_drop, W3) + b3)

    # Initialise variables
    sess.run(tf.initialize_all_variables())

    # Define cost function
    cross_entropy = - tf.reduce_sum(y_ * tf.log(y))

    # Train
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    for i in range(20000):
        batch = mnist.train.next_batch(50) # Load 50 training examples

        # Display progress
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    vars_to_save = {}

    vars_to_save['W1'] = sess.run(W1).tolist()
    vars_to_save['b1'] = sess.run(b1).tolist()
    vars_to_save['W2'] = sess.run(W3).tolist()
    vars_to_save['b2'] = sess.run(b3).tolist()

    open("mnist_cnn.json", "w").write(json.dumps(vars_to_save, indent=2))


if __name__ == '__main__':
    main()
