# -*- coding:utf-8 -*-
# 编辑 : frost
# 时间 : 2020/4/18 18:24
import cv2 as cv
import numpy as np


#  保存图片数据
index = 1
drawing = False
ix , iy = -1, -1
#  实在不行的话，直接封装成一个函数
def draw(event, x, y, flags, params):
    b = g = r = 255
    color = (b, g, r)
    global drawing, ix, iy
    #  点击鼠标开始绘制
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    #  鼠标左键按住的情况下面，鼠标移动
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        if drawing == True:
            cv.circle(img, (x, y), 14, color, -1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing == False


img = np.zeros((360, 360, 3), np.uint8)
cv.namedWindow("")
cv.setMouseCallback("", draw)
while True:
    cv.imshow("", img)
    k = cv.waitKey(1)
    if k == 27:
        break
    elif k == 32:
        img = np.zeros((360, 360, 3), np.uint8)
    elif k == 115:
        cv.imwrite("./pictures/test_{}.jpg".format(index), img)
        index += 1