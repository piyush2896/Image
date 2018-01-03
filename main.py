from util import Config, read_and_decode
import tensorflow as tf
import numpy as np
from visualize import plot_loss_and_accuracy
import model
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(n_iter=20, activation_type=model.USE_RELU):
    """
    Train Model for given iterations and based on different activation function.
    @params
        n_iter: Number of Epochs
        activation_type: Type of Activation Used. Possible Values => model.USE_RELU or model.USE_LEAKY_RELU
    @returns
        hist: A dictionary of training history of the form ->
            {
                'train': {
                    'loss': [loss_1, loss_2,... n_iter],
                    'acc': [acc_1, acc_2,... n_iter]
                },
                'dev': {
                    'loss': [loss_d_1, loss_d_2,... n_iter],
                    'acc': [acc_d_1, acc_d_2,... n_iter]
                }
            }
    """
    tf.set_random_seed(2)
    model_, params = model.load_model(activation_type=activation_type)

    X = model_['input']
    Y_hat = tf.nn.sigmoid(model_['out'], name='pred')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='true_value')

    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_['out'], labels=Y))

    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

    hist = {
        'train': {'loss': [], 'acc': []}, 
        'dev': {'loss':[], 'acc': []}
    }

    saver = tf.train.Saver()
    train_queue = tf.train.string_input_producer([Config.TFRECORD_TRAIN],
                                                 num_epochs=(n_iter+1) * 157,
                                                 shuffle=True)
    dev_queue = tf.train.string_input_producer([Config.TFRECORD_DEV],
                                               num_epochs=(n_iter+1)*20,
                                               shuffle=True)

    with tf.Session() as sess:
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img_train, label_train = read_and_decode(train_queue)
        img_dev, label_dev = read_and_decode(dev_queue)
        for i in range(n_iter):

            t_loss, t_acc, d_loss, d_acc = 0, 0, 0, 0

            for j in range(150):
                img, label = sess.run([img_train, label_train])
                _, l, y_hat, y = sess.run([train_step, loss, Y_hat, Y],
                                          feed_dict={X: img, Y: label})
                t_loss += l
                t_acc += int((y_hat >= 0.5) == y)

            hist['train']['loss'].append(t_loss / 150)
            hist['train']['acc'].append(t_acc / 150)

            for j in range(15):
                img, label = sess.run([img_dev, label_dev])
                l, y_hat, y = sess.run([loss, Y_hat, Y],
                                       feed_dict={X: img, Y: label})
                d_loss += l
                d_acc += int((y_hat >= 0.5) == y)

            hist['dev']['loss'].append(d_loss / 15)
            hist['dev']['acc'].append(d_acc / 15)

            print('\nEpoch: {}-----------------'.format(i))
            print('Loss: {}\tAcc: {}'.format(round(hist['train']['loss'][-1], 5),
                                                        round(hist['train']['acc'][-1], 5)))
            print('ValLoss: {}\tVal Acc: {}'.format(round(hist['dev']['loss'][-1], 5),
                                                    round(hist['dev']['acc'][-1], 5)))

            # Override old model with every 10th model
            if i % 10 == 0:
                if activation_type == model.USE_RELU:
                    saver.save(sess, Config.NORMAL_MODEL_PATH)
                else:
                    saver.save(sess, Config.LEAKY_MODEL_PATH)

        # Save the Last Model
        if activation_type == model.USE_RELU:
            saver.save(sess, Config.NORMAL_MODEL_PATH)
        else:
            saver.save(sess, Config.LEAKY_MODEL_PATH)
        print('Final Model Saved!!')

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
    return hist


def test(activation_type=model.USE_RELU):
    model_, params = model.load_model(activation_type=activation_type)

    X = model_['input']
    Y_hat = tf.nn.sigmoid(model_['out'], name='pred')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='true_value')

    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_['out'], labels=Y))

    saver = tf.train.Saver()

    test_queue = tf.train.string_input_producer([Config.TFRECORD_TEST])

    with tf.Session() as sess:
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if activation_type == model.USE_RELU:
            saver.restore(sess, Config.NORMAL_MODEL_PATH)
        else:
            saver.restore(sess, Config.LEAKY_MODEL_PATH)

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img_test, label_test = read_and_decode(test_queue)

        t_loss, t_acc = 0, 0
        for i in range(20):
            image, label = sess.run([img_test, label_test])
            l, y, y_hat = sess.run([loss, Y, Y_hat], feed_dict={X:image, Y: label})
            t_acc += int((y_hat >= 0.5) == y)
            t_loss += l

        t_acc = t_acc / 21
        t_loss = t_loss / 21

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)

    print("Test Accuracy: {}\tTest Loss: {}".format(round(t_acc, 5), round(t_loss, 5)))


def run():
    print("Training Model with Simple Relu Activation.")
    hist_non_leaky = train()
    tf.reset_default_graph()

    print("\n\nTraining Model with Custom Leaky Relu Activation.")
    hist_leaky = train(activation_type=model.USE_LEAKY_RELU)
    plot_loss_and_accuracy(hist_non_leaky, hist_leaky)

    print("\n\nTesting Model with Simple Relu Activation.")
    tf.reset_default_graph()
    test()

    print("\n\nTesting Model with Custom Leaky Relu Activation.")
    tf.reset_default_graph()
    test(activation_type=model.USE_LEAKY_RELU)

if __name__ == '__main__':
    run()
