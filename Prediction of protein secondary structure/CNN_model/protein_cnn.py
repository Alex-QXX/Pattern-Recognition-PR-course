import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

#  定义网络结构
#  读取训练和验证集数据，在data_processing中已经将数据处理后存成.npy格式
X_train = np.load('./train_X.npy')
y_train = np.load('./train_y.npy')
X_validation = np.load('./val_X.npy')
y_validation = np.load('./val_X.npy')

# 定义网络的输出与输出变量
X = tf.placeholder(tf.float32, [None, 100, 22], name="X")
y = tf.placeholder(tf.float32, [None, 100, 4], name='y')

# 模型
filter_width = 11  # 卷积核尺寸
filter_input_size = 22  # 输入数据是22维的
filter_channels = 40    # 卷积核通道数
final_filter_channels = 4   # 输出通道数应为对应的分类种类
# 定义每一层卷积的卷积核参数
W1 = tf.get_variable(name="W1", shape=[filter_width, filter_input_size, filter_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
b1 = tf.get_variable("b1",[filter_channels], initializer=tf.random_normal_initializer())
W2 = tf.get_variable(name="W2", shape=[filter_width, filter_input_size, filter_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
b2 = tf.get_variable("b2",[filter_channels], initializer=tf.random_normal_initializer())
W3 = tf.get_variable(name="W3", shape=[filter_width, filter_input_size, final_filter_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
b3 = tf.get_variable("b3",[final_filter_channels], initializer=tf.random_normal_initializer())

# 卷积层1
conv1 = tf.nn.conv1d(value=X, filters=W1, stride=1, padding='SAME')
a1 = tf.nn.bias_add(conv1, b1)
z1 = tf.nn.relu(a1)
# 卷积层2
conv2 = tf.nn.conv1d(value=X, filters=W2, stride=1, padding='SAME')
a2 = tf.nn.bias_add(conv2, b2)
z2 = tf.nn.relu(a2)
# 卷积层3
conv3 = tf.nn.conv1d(value=X, filters=W3, stride=1, padding='SAME')
a3 = tf.nn.bias_add(conv3, b3)
z3 = tf.nn.relu(a3)
# 通过softmax输出类别
y_ = tf.nn.softmax(z3)

mask = tf.not_equal(tf.argmax(y, 2), 3)

y_masked = tf.boolean_mask(y, mask)
z3_masked = tf.boolean_mask(z3, mask)
y__masked = tf.boolean_mask(y_, mask)

# loss(交叉熵函数)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_masked, logits=z3_masked))

# 学习率
learning_rate = tf.placeholder(tf.float32)

# Adam优化算法进行梯度下降最小化loss
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# 输出的预测值与准确率的定义
prediction = tf.argmax(y__masked, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_masked, 1)), tf.float32))

# 初始化条件
init = tf.global_variables_initializer()

n_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print("Number of parameters:", n_parameters)


# Start as session
with tf.Session() as session:

    batch_size = 100
    # 先对模型的参数初始化
    session.run(init)
    # 每一轮训练
    for epoch in range(100):
        print("Epoch:", epoch)
        # 每次输入100个样本 batch_size=100 学习率 learning_rate=0.01
        for b in range(0, X_train.shape[0], batch_size):
            _, loss_value = session.run([optimizer, loss], feed_dict={X: X_train[b:b+batch_size],
                                                                      y: y_train[b:b+batch_size],
                                                                      learning_rate: 0.01})
            if b % 1000 == 0:
                validation_accuracy = session.run(accuracy, feed_dict={X: X_validation, y: y_validation})
                print("loss[b=%04d] = %f, val_acc = %f" % (b, loss_value, validation_accuracy))

    print("训练完毕")

    # 计算准确率
    train_accuracy_value, pred_train = session.run([accuracy, prediction], feed_dict={X: X_train, y: y_train})
    print("Train accuracy:", train_accuracy_value)

    val_accuracy_value, pred_val = session.run([accuracy, prediction], feed_dict={X: X_validation, y: y_validation})
    print("Val accuracy:", val_accuracy_value)
