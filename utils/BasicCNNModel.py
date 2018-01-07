from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) #the truncated_normal is creating a gaussian distributed random sampling?

  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2_same(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def basicCNNModel(x_train, y_train):
#mnist = input_data.read_data_sets("mnist dataset/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 90, 90, 3])
    y_ = tf.placeholder(tf.float32, [None, 120])

    #conv1 + pooling1
    #kernel size [5, 5, 3, 32]
    W_conv1 = weight_variable([5, 5, 3, 16])
    b_conv1 = bias_variable([16])   #bias is wrt to one image

    x_image = tf.reshape(x, [-1, 90, 90, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2_same(h_conv1)
    # the activation size is now 45*45*16

    #conv2 + pooling2
    W_conv2 = weight_variable([5, 5, 16, 16])
    b_conv2 = bias_variable([16])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #size [45,45,64]
    h_pool2 = max_pool_2x2_same(h_conv2)
    #the activation size is now [23,23,16]

    #conv3 + pooling3
    W_conv3 = weight_variable([5, 5, 16, 32])
    b_conv3 = bias_variable([32])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) #size [23,23,32]
    h_pool3 = max_pool_2x2_same(h_conv3)
    #the activation size is now 12*12*32


    # conv4 + pooling4
    W_conv4 = weight_variable([5, 5, 32, 32])
    b_conv4 = bias_variable([32])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)  # size [12,12,32]
    h_pool4 = max_pool_2x2_same(h_conv4)
    # the activation size is now 6*6*32

    #densely connected layer
    W_fc1 = weight_variable([6 * 6 * 32, 512])
    b_fc1 = bias_variable([512])

    h_pool4_flat = tf.reshape(h_pool4, [-1, 6*6*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    #drop out, drop out should be used for the whole network? for this example it is only used for one layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #readout layer
    W_fc2 = weight_variable([512, 120])
    b_fc2 = bias_variable([120])

    y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name= "logits")

    #training
    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name= "accuracy")

    idx = 0
    #adding model saving
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            #batch = x_train.next_batch(50)

            if idx + 50 > 10222:
                x_batch = x_train[idx:,:,:,:]
                y_batch = y_train[idx:, :]
                idx = 0
            else:
                x_batch = x_train[idx:(idx+50), :, :, :]
                y_batch = y_train[idx:(idx+50), :]
                idx = idx + 50

            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                  x: x_batch, y_: y_batch, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                save_path = saver.save(sess, "models/basic CNN/model.ckpt")
            if i % 300 == 0:
                restore_sess = tf.Session()
                saver = tf.train.import_meta_graph('models/basic CNN/model.ckpt.meta')
                saver.restore(restore_sess, tf.train.latest_checkpoint('./models/basic CNN/'))
                graph = tf.get_default_graph()
                accuracy_restore = graph.get_tensor_by_name("accuracy:0")
                logits = graph.get_tensor_by_name("logits:0")
                print("training accuracy from restored model in step %d is %g" % (
                i, restore_sess.run(accuracy_restore, feed_dict={
                    x: x_batch, y_: y_batch, keep_prob: 1.0})))
                print("logits shape is:" ,
                    restore_sess.run(tf.shape(logits), feed_dict={
                        x: x_batch, y_: y_batch, keep_prob: 1.0}))


            train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})

                #print('test accuracy %g' % accuracy.eval(feed_dict={
                #x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))