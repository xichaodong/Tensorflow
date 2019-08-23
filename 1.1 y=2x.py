import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # 从-1到1中间随机生成100个数
    train_x = np.linspace(-1, 1, 100)
    # 将x乘2并每一个加上一个随机值再*0.3作为干扰噪声
    train_y = train_x * 2 + np.random.randn(*train_x.shape) * 0.3

    # 在计算图中用占位符定义x
    X = tf.placeholder("float")
    # 在计算图中用占位符定义y
    Y = tf.placeholder("float")

    # 在计算图中定义权重w
    W = tf.Variable(tf.random_normal([1]), name="weight")
    # 在计算图中定义偏置b
    b = tf.Variable(tf.zeros([1]), name="bias")

    # 权重矩阵和输入矩阵相乘再加上偏置
    z = tf.multiply(X, W) + b

    # 计算损失函数的输出（方差）
    cost = tf.reduce_mean(tf.square(Y - z))
    # 定义梯度下降学习率
    learn_rate = 0.01
    # 梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    # 初始化计算图中的变量
    init = tf.global_variables_initializer()
    # 定义迭代次数
    training_epochs = 20
    # 定义打印步长
    display_step = 2

    # 开启session
    with tf.Session() as sess:
        # 初始化
        sess.run(init)
        # 开始迭代
        for epoch in range(training_epochs):
            # 用zip把x和y按元素位置打包
            for(x, y) in zip(train_x, train_y):
                # 执行一次反向传播
                sess.run(optimizer, feed_dict={X: x, Y: y})
            # 如果应该打印
            if epoch % display_step == 0:
                # 打印损失函数的输出和变量信息
                loss = sess.run(cost, feed_dict={X: train_x, Y: train_y})
                print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
        print("Finished")
        print("cost=", sess.run(cost, feed_dict={X: train_x, Y: train_y}), "W=", sess.run(W), "b=", sess.run(b))
        # 测试训练结果
        print("x=0.2, z=", sess.run(z, feed_dict={X: 0.2}))
