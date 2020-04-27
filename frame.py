# -*- coding:utf-8 -*-
# 编辑 : frost
# 时间 : 2020/4/17 17:53
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import END
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL import Image,ImageTk
from skimage import io,color,transform

# 全局变量
get_image = None
show_image = None

# 构造函数
class GetTk(object):
    """ 不继承 """
    #  主窗口
    root = tk.Tk()
    #  按钮
    button1 = button2 = button3 = button4 = None
    #  画布
    canvas = None
    #  提示信息
    test_information = None
    test_var = None
    choice_image = None
    use_information = None
    #  标题
    label_of_canvas = None
    label_of_text = None

    #  主界面
    def set_windows(self):
        self.root.title("手写数字识别")
        self.root.resizable(False, False)
        self.root.geometry("1200x600")
        # return self.root

    #  循环刷新界面  没有使用对象里面的参数，所以需要设置成为静态方法
    @staticmethod
    def set_loop():
        tk.mainloop()

    #  按键
    def set_button1(self):
        self.button1 = tk.Button(self.root, text="绘制", font=("华文楷书", 13),
                                 bg="#E8D098", fg="blue", width=10, height=1,
                                 padx=1, pady=2, command=self.draw_pic)
        self.button1.place(x=630, y=480)

    def set_button3(self):
        self.button3 = tk.Button(self.root, text="识别", font=("华文楷书", 13),
                                 bg="#E8D098", fg="blue", width=10, height=1,
                                 padx=1, pady=2, command=self.insert_text)
        self.button3.place(x=820, y=480)

    def set_button4(self):
        self.button4 = tk.Button(self.root, text="清屏", font=("华文楷书", 13),
                                 bg="#E8D098", fg="blue", width=10, height=1,
                                 padx=1, pady=2, command=self.cls)
        self.button4.place(x=1000, y=480)

    #  画布
    def set_canvas(self):
        self.canvas = tk.Canvas(self.root, bg='white', width=80, height=80)
        self.canvas.place(x=500, y=360)

    #  提示信息
    def set_text(self):
        self.test_information = ScrolledText(self.root, fg="blue", font=("华文楷书", 10), width=60, height=21)
        self.test_information.place(x=600, y=105)

    #  信息数据的变更
    def change_information(self):
        self.test_var = tk.StringVar()
        self.test_var = "测试数据"
        return self.test_var

    #  标签
    def set_choice(self):
        self.choice_image = tk.Label(self.root, text="待识别图片", font=("华文楷书", 11), fg="blue")
        self.choice_image.place(x=495, y=340)

    def set_label_canvas(self):
        self.label_of_text = tk.Label(self.root, text="使用说明", font=("华文楷书", 20), fg="purple")
        self.label_of_text.place(x=200, y=60)

    def set_use_information(self):
        information =  "1. 点击“绘制”按钮，在弹出的画布上绘制待识别数字。\n"
        information2 = "2. 使用鼠标左键绘制图画。点击‘s’,保存当前图片;\n"
        information3 = "点击“SPACE”,刷新画布;点击”ESC“,退出绘制界面。\n"
        information4 = "3. 点击”识别“按钮,在弹出的窗口中选择目标图片。\n"
        information5 = "4. 点击”清屏“按钮,将程序界面数据清空。\n"
        self.use_information = tk.Label(self.root, text=information, font=("华文楷书", 12), fg="green")
        tk.Label(self.root, text=information2, font=("华文楷书", 12), fg="green" ).place(x=40, y=170)
        tk.Label(self.root, text=information3, font=("华文楷书", 12), fg="green" ).place(x=40, y=200)
        tk.Label(self.root, text=information4, font=("华文楷书", 12), fg="green" ).place(x=40, y=230)
        tk.Label(self.root, text=information5, font=("华文楷书", 12), fg="green" ).place(x=40, y=260)
        self.use_information.place(x=40, y=140)

    def set_labels_text(self):
        self.label_of_text = tk.Label(self.root, text="处理信息", font=("华文楷书", 20), fg="purple")
        self.label_of_text.place(x=800, y=60)

    def show_on_canvas(self, the_path):
        print(the_path)
        global  show_image,get_image
        get_image = Image.open(the_path)
        get_image = get_image.resize((80,80))
        show_image = ImageTk.PhotoImage(get_image)
        # 改变图片大小，进行展示
        self.canvas.create_image(40, 40, image=show_image)

    #  绑定插入数据
    def insert_text(self):
        #  打开文件夹
        try:
            get_path = tk.filedialog.askopenfilename()
            if get_path:
                information = '您选择了新的图片,所在位置是 : '
                self.test_information.tag_config("tag_1", backgroun="yellow", foreground="red")
                self.test_information.tag_config("tag_2", backgroun="white", foreground="blue")
                self.test_information.insert("end", information, "tag_2")
                self.test_information.insert("insert", get_path, "tag_1")
                self.test_information.insert("insert", "\n")
                self.test_information.see(END)
                # 在画布上面显示图片
                self.show_on_canvas(get_path)
                #  识别
                #  直接采用PIL处理图像
                # get_image = Image.open(get_path).convert('L')
                # im = get_image.resize((28, 28))
                # im = np.asarray(im)

                # im = cv.imread(get_path)  # 读取图片
                # im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)  # 转换了灰度化
                # im = cv.resize(im, (28, 28), interpolation=cv.INTER_NEAREST)

                # 图像预处理采用第三种方法
                im = io.imread(get_path)
                im = color.rgb2gray(im)
                im = transform.resize(im, (28,28))

                im = im.reshape((-1, 784))
                im = im.astype('float')
                the_x = im
                the_y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 这里这个不是关键，所以可以随便给初始值
                the_y = the_y.reshape([1, 10])

                with tf.Session() as sess:
		    # 打包的时候写成绝对路径
                    new_saver = tf.train.import_meta_graph('F:/digits/CNN/checkpoint-2/digits.ckpt-20000.meta')
                    get_model = new_saver.restore(sess, 'F:/digits/CNN/checkpoint-2/digits.ckpt-20000')
                    x = sess.graph.get_tensor_by_name("input_x:0")
                    result = sess.graph.get_tensor_by_name("predict:0")
                    result = sess.run(result, feed_dict={x: the_x})
                    # print(np.argmax(result))
                    information = '该图片的识别结果为 :'
                    self.test_information.insert("end", information, "tag_2")
                    self.test_information.insert("insert", np.argmax(result), "tag_1")
                    self.test_information.insert("insert", "\n")
                    self.test_information.see(END)
            else:
                print("用户取消")
        except:
            print("您没有选择图片")

    def cls(self):
        self.test_information.delete(0.0, END)
        # 将带显示图片清空
        self.set_canvas()


    def draw_pic(self):
        index = 1
        drawing = False
        ix, iy = -1, -1

        def draw(event, x, y, flags, image):
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

        #  实在不行的话，直接封装成一个函数
        img = np.zeros((360, 360, 3), np.uint8)
        cv.namedWindow("")
        cv.setMouseCallback("", draw)
        while True:
            cv.imshow("", img)
            k = cv.waitKey(1)
            if k == 27:
                cv.destroyAllWindows()
                break
            elif k == 32:
                img = np.zeros((360, 360, 3), np.uint8)
            elif k == 115: #  这里是小写字母
                cv.imwrite("F:/digits/CNN/pictures/test_{}.jpg".format(index), img)
                index += 1

    #  监听案件
    def listen_button3(self):
        self.insert_text()

    #  清空屏幕
    def listen_button4(self):
        self.cls()

    #  绘制图画
    def listen_button1(self):
        self.draw_pic()


if __name__ == '__main__':
    print("绘制主界面")
    win = GetTk()
    win.set_windows()
    #  添加组件
    win.set_button1()
    # win.set_button2()
    win.set_button3()
    win.set_button4()
    #  画布 和 提示框
    win.set_use_information()
    win.set_canvas()
    win.set_text()
    win.set_choice()
    #  标题
    win.set_label_canvas()
    win.set_labels_text()
    win.set_loop()
