from ResNeXt import *
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(1, num_epochs + 1):
        if epoch == (num_epochs * 0.5) or epoch == (num_epochs * 0.75):
            lr = lr / 10

        cur = 0

        print('start')
        for i in range(1, iteration + 1):
            if cur + batch_size < 50000:
                batch_x = x_train[cur: cur + batch_size]
                batch_y = y_train[cur: cur + batch_size]
            else:
                batch_x = x_train[cur:]
                batch_y = y_train[cur:]

            batch_x = data_augmentation(batch_x)

            sess.run([train], feed_dict={x: batch_x, y: batch_y, learning_rate: lr, flag: True})

            cur += batch_size

            print(i)

        test_acc = 0.0
        test_cur = 0
        test_batch = 10000 // test_iteration

        for it in range(test_iteration):
            test_batch_x = x_test[test_cur: test_cur + test_batch]
            test_batch_y = y_test[test_cur: test_cur + test_batch]
            test_cur = test_cur + test_batch
            acc_ = sess.run(accuracy,
                            feed_dict={x: test_batch_x, y: test_batch_y, learning_rate: lr,
                                       flag: False})
            test_acc += acc_

        test_acc /= test_iteration  # average accuracy

        print("epoch: %d/%d, test_acc: %.4f \n" % (epoch, num_epochs, test_acc))
