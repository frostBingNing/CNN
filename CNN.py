# -*- coding:utf-8 -*-
# 编辑 : frost
# 时间 : 2020/4/9 1:27
# 晚上开始编写卷积神经网络的经典模型 LeNet-5 模型  ---   提高准确度  理解本质才是关键
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np
import os

#  屏蔽警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#  加载数据
mnist_data_set = input_data.read_data_sets("MNIST_data/", one_hot=True)

#  声名图片类型
x = tf.placeholder('float', [None, 784], name='input_x')
y_ = tf.placeholder('float', [None, 10])
#  输入的图片数据转化
x_image = tf.reshape(x, [-1, 28, 28, 1])

#  第一层卷积层的设计
filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]), name='filter1')
bias1 = tf.Variable(tf.truncated_normal([6]), name='bias1')
conv1 = tf.nn.conv2d(x_image, filter1, strides=[1, 1, 1, 1], padding='SAME')
r_conv1 = tf.nn.sigmoid(conv1 + bias1)
# r_conv1 = tf.nn.relu(conv1 + bias1)


#  第二层 池化操作    池化全部采用 2*2
max_pool2 = tf.nn.max_pool(r_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#  第三层 卷积设计
filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]), name='filter2')
bias2 = tf.Variable(tf.truncated_normal([16]), name='bias2')
conv2 = tf.nn.conv2d(max_pool2, filter2, strides=[1, 1, 1, 1], padding='SAME')
r_conv2 = tf.nn.sigmoid(conv2 + bias2)

#  第四层  池化操作
max_pool3 = tf.nn.max_pool(r_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#  第五层  卷积生成120个
filter3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120]), name='filter3')
bias3 = tf.Variable(tf.truncated_normal([120]), name='bias3')
conv3 = tf.nn.conv2d(max_pool3, filter3, strides=[1, 1, 1, 1], padding='SAME')
r_conv3 = tf.nn.sigmoid(conv3 + bias3)
# r_conv3 = tf.nn.relu(conv3 + bias3)

#  全连接层 第六七层  --  分类器的作用
w1 = tf.Variable(tf.truncated_normal([7 * 7 * 120, 80]), name='w1')
b1 = tf.Variable(tf.truncated_normal([80]), name='b1')
r_pool_flat = tf.reshape(r_conv3, [-1, 7 * 7 * 120])
fc1 = tf.nn.sigmoid(tf.matmul(r_pool_flat, w1) + b1)
# fc1 = tf.nn.relu(tf.matmul(r_pool_flat, w1) + b1)

#  最后一层 分类  softmax
w2 = tf.Variable(tf.truncated_normal([80, 10]), name='w2')
b2 = tf.Variable(tf.truncated_normal([10]), name='b2')
y_conv = tf.nn.softmax(tf.matmul(fc1, w2) + b2, name="predict")

#  损失函数
cross = -tf.reduce_sum(y_ * tf.log(y_conv))
#  梯度下降法
trian_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross)
sess = tf.InteractiveSession()

#  测试正确率
correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

#  所有变量初始化
# sess.run(tf.initialize_all_variables())

#  用来保存模型
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

#  开始训练
start_time = time.time()

sess.run(init_op)
# 迭代20000此看一下效果
for i in range(1, 20001):
    #  获取训练数据
    batch_x, batch_y = mnist_data_set.train.next_batch(200)

    #  每当迭代2个batch 进行测试，输出预测准确率
    #  其实这中间仅仅是一种保存参数和参数输出的过程
    if i % 2 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_x, y_: batch_y
        })
        train_loss = cross.eval(feed_dict={
            x: batch_x, y_: batch_y
        })
        print("step %d, training accuracy %g, loss is %g" % (i, train_accuracy, train_loss))
        #  时间间隔
        end_time = time.time()
        print("time : ", (end_time - start_time))
        start_time = end_time
    #  每10此保存一次
    if i % 100 == 0:
        saver.save(sess, "./checkpoint-2/digits.ckpt", global_step=i)
    #  采用梯度下降发进行训练
    trian_step.run(feed_dict={
        x: batch_x, y_: batch_y
    })

#  准确率 0.95左右 还可以，接下来就是调用此模型
#  手写数字识别率较高的是 0 2 3 1 5 4 7  较低的是 8 9 6