# -*- coding:utf-8 -*-
# 编辑 : frost
# 时间 : 2020/4/22 16:51

# 该文件调用模型进行测试  加载-输入-预测处理  最后进行整合处理
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import array
import cv2 as cv

#  屏蔽警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# index = int(input())

mnist_data_set = input_data.read_data_sets("MNIST_data/", one_hot=True)


#  需要增加二值化 归一化 处理  最后将图片尽量放置在中心位置 ----  现在的主要目标就是将自己的图片转化为那种格式，理论上提升识别的准确率

# the_x = mnist_data_set.test.images[index]
# the_x = the_x.reshape([1, 784])
#  更新the_x


# 测试test数据集的准确率问题
im = cv.imread('F:/network/CNN/pictures/test_1.jpg')  #读取图片
im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)   #转换了灰度化
im = cv.resize(im, (28,28), interpolation=cv.INTER_NEAREST)
im = im.reshape((-1, 784))
im = im.astype('float')
the_x = im


the_y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 这里这个不是关键，所以可以随便给初始值
the_y = the_y.reshape([1, 10])


# 调用训练好的模型
with tf.Session() as sess:
    # 测试test的准确率
    new_saver = tf.train.import_meta_graph('./checkpoint-2/digits.ckpt-20000.meta')
    get_model = new_saver.restore(sess, './checkpoint-2/digits.ckpt-20000')
    x = sess.graph.get_tensor_by_name("input_x:0")
    result = sess.graph.get_tensor_by_name("predict:0")
    count = 0
    for i in range(10000):
        # 测试4000张
        the_x = mnist_data_set.test.images[i].reshape((-1,784))
        true_result = np.argmax(mnist_data_set.test.labels[i])
        result_1 = sess.run(result, feed_dict={x: the_x})
        print(np.argmax(result_1), true_result)
        if(true_result == np.argmax(result_1)):
            count += 1
        # 准确率
        print("the count is{}".format(i))
        print("the model of test images's accuracy is {}".format(count/(i+1)))

# print(mnist_data_set.test.labels[index])
