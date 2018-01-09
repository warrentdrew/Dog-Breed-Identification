#this file converts an tensorflow tensor to a CSV file
#as an example, the file now restore data from a ckpt data and convert them into CSV

import tensorflow as tf
import numpy as np
import pandas as pd


def Submission(x_test):
    #x = tf.placeholder(tf.float32, [None, 90, 90, 3])
    #y_ = tf.placeholder(tf.float32, [None, 120])
    #keep_prob = tf.placeholder(tf.float32)

    a = np.zeros([100, 90, 90, 3])
    b = np.zeros([100, 120])



    restore_sess = tf.Session()

    saver = tf.train.import_meta_graph('./models/basic CNN/model.ckpt.meta')
    saver.restore(restore_sess, tf.train.latest_checkpoint('./models/basic CNN/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    feed_dict = {x: x_test, keep_prob: 1.0}
    #accuracy_restore = graph.get_tensor_by_name("accuracy:0")
    logits = graph.get_tensor_by_name("logits:0")
    results = tf.nn.softmax(logits=logits)

    #print("logits shape is:" , restore_sess.run(tf.shape(logits), feed_dict=feed_dict))

    df_test = pd.read_csv('../input/sample_submission.csv', header= None)
    df_test.values[1:,1:] = results.eval(session=restore_sess, feed_dict=feed_dict)

    print(df_test.values.shape)
    df_test.to_csv("./submissions/submission_basicCNN.csv", index = False, header = False)


