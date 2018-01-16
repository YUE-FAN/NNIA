from ResNeXt import *
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    epoch_learning_rate = init_learning_rate
    for epoch in range(1, total_epochs + 1):
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0

        print('start')
        for step in range(1, iteration + 1):
            if pre_index + batch_size < 50000:
                batch_x = train_x[pre_index: pre_index + batch_size]
                batch_y = train_y[pre_index: pre_index + batch_size]
            else:
                batch_x = train_x[pre_index:]
                batch_y = train_y[pre_index:]

            batch_x = data_augmentation(batch_x)

            sess.run([train], feed_dict={x: batch_x, label: batch_y, learning_rate: epoch_learning_rate, training_flag: True})

            pre_index += batch_size

            # print(step)

        test_acc = 0.0
        test_pre_index = 0
        add = 10000 // test_iteration

        for it in range(test_iteration):
            test_batch_x = test_x[test_pre_index: test_pre_index + add]
            test_batch_y = test_y[test_pre_index: test_pre_index + add]
            test_pre_index = test_pre_index + add
            acc_ = sess.run(accuracy,
                            feed_dict={x: test_batch_x, label: test_batch_y, learning_rate: epoch_learning_rate,
                                       training_flag: False})
            test_acc += acc_

        test_acc /= test_iteration  # average accuracy

        print("epoch: %d/%d, test_acc: %.4f \n" % (epoch, total_epochs, test_acc))
