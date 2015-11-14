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

    # Define weights and bias
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    # Initialise variables
    sess.run(tf.initialize_all_variables())

    # Define prediction
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # Define cost function
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    # Train
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    for i in range(1000):
        batch = mnist.train.next_batch(50) # Load 50 training examples
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # Evaluate model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    main()
