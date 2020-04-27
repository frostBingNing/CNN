# -*- coding:utf-8 -*-
# 编辑 : frost
# 时间 : 2020/4/27 22:35


from skimage import io,transform,color
from matplotlib import pyplot as plt
#  处理图片的第三方库函数

the_image_path = "./pictures/test_2.jpg"
img = io.imread(the_image_path)
img = color.rgb2gray(img)
# 放缩图片 然后2828
img = transform.resize(img,(28, 28))
print(img.shape, img.dtype.name)

plt.subplot(1,1,1)
plt.imshow(img)