import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    # 加载mnist数据集
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 在计算图中用占位符定义x (None表示第一维不确定，第二维图像像素784 = 32 * 32)
    x = tf.placeholder(tf.float32, [None, 784])
    # 在计算图中用占位符定义y  (None表示第一维不确定，第二维一共十个数字，所以one_hot维数为10)
    y = tf.placeholder(tf.float32, [None, 10])

    # 在计算图中定义权重，随机生成一个784行10列的矩阵，这样与x相乘之后就相当于为784个像素每个设置了一个变量，10对应10个数字
    w = tf.Variable(tf.random_normal([784, 10]))
    # 在计算图中定义偏置，这里就定义成0
    b = tf.Variable(tf.zeros([10]))

    # 输入乘以权重后再加上偏置，然后用SoftMax激活
    pred = tf.nn.softmax(tf.matmul(x, w) + b)
    # 计算每个计算值和实际值的交叉熵， 然后取平均值作为损失函数的输出
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

    # 设置学习率为0.01
    learn_rate = 0.01
    # 设置梯度优化器
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    # 定义迭代次数
    training_epochs = 25
    # 设置mini_batch过程每次取的数量
    batch_size = 100
    # 定义打印步长
    display_step = 1

    # 开启session
    with tf.Session() as sess:
        # 初始化计算图中变量
        sess.run(tf.global_variables_initializer())
        # 开始迭代
        for epoch in range(training_epochs):
            # 定义平均损失
            avg_cost = 0
            # 计算总共需要取的次数
            total_batch = int(mnist.train.num_examples / batch_size)
            # 开始迭代
            for i in range(total_batch):
                # 取一批数据
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # 将x， y注入计算图，执行一次反向传播
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                # 更新平均损失
                avg_cost += c / total_batch
            # 如果应该打印
            if (epoch + 1) % display_step == 0:
                # 打印当前训练轮数和平均损失
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        # 对比模型输出与测试集的label
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        # 计算正确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 打印正确率
        print("Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
        print("Finished")
