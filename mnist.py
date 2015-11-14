import input_data
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def inference(x, y_):
    # Reshape input to apply the first layer
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First layer of CNN
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second layer of CNN
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Third layer (Densely connected)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Apply dropout to prevent overfitting
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Fourth layer (readout layer / softmax layer)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

def loss(y_, y_conv):
    cross_entropy = - tf.reduce_sum(y_ * tf.log(y_conv))
    return cross_entropy

def train(loss):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return train_step

def evaluate(y_, y_conv):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

def main():
    # Load data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Start TensorFLow session
    sess = tf.InteractiveSession()

    # Define input / output
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    # Infere
    y_conv = inference(x, y_)

    # Define cost function
    cost_func = loss(y_, y_conv)

    # Train
    train_step = train(cost_func)

    # Evaluate
    accuracy = evaluate(y_, y_conv)

    # Start session
    sess.run(tf.initialize_all_variables())

    # Training loop
    for i in range(20000):
        # Train over 50 training examples per iteration
        batch = mnist.train.next_batch(50)

        # Display progress
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                      y_: batch[1],
                                                      keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # Show final results
    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images,
                                                        y_: mnist.test.labels,
                                                        keep_prob: 1.0}))


if __name__ == '__main__':
    main()
