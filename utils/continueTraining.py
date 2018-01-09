import tensorflow as tf

def continueTrain(x_train, y_train):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./models/basic CNN/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./models/basic CNN/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    accuracy = graph.get_tensor_by_name("accuracy:0")
    cross_entropy = graph.get_tensor_by_name("cross_entropy:0")
    train_step = graph.get_operation_by_name("train_step")

    idx = 0
    with sess:
        for i in range(25000):
            # batch = x_train.next_batch(50)

            if idx + 50 > 10222:
                x_batch = x_train[idx:, :, :, :]
                y_batch = y_train[idx:, :]
                idx = 0
            else:
                x_batch = x_train[idx:(idx + 50), :, :, :]
                y_batch = y_train[idx:(idx + 50), :]
                idx = idx + 50

            if i % 100 == 0:
                train_accuracy = accuracy.eval( feed_dict={
                    x: x_batch, y_: y_batch, keep_prob: 1.0})
                train_loss = cross_entropy.eval( feed_dict={
                    x: x_batch, y_: y_batch, keep_prob: 1.0})
                print('step %d, training accuracy %g, training loss %g' % (i, train_accuracy, train_loss))
                save_path = saver.save(sess, "models/basic CNN/model.ckpt")
            # if i % 300 == 0:
            #     # restore_sess = tf.Session()
            #     # saver = tf.train.import_meta_graph('models/basic CNN/model.ckpt.meta')
            #     # saver.restore(restore_sess, tf.train.latest_checkpoint('./models/basic CNN/'))
            #     # graph = tf.get_default_graph()
            #     # accuracy_restore = graph.get_tensor_by_name("accuracy:0")
            #     # logits = graph.get_tensor_by_name("logits:0")
            #     print("training accuracy from restored model in step %d is %g" % (
            #     i, restore_sess.run(accuracy_restore, feed_dict={
            #         x: x_batch, y_: y_batch, keep_prob: 1.0})))
            #     # print("logits shape is:" ,
            #     #    restore_sess.run(tf.shape(logits), feed_dict={
            #     #        x: x_batch, y_: y_batch, keep_prob: 1.0}))


            train_step.run( feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})