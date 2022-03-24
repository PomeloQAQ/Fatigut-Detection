import sys
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pyttsx3 # 语音播报模块
from collections import OrderedDict # 导入顺序表
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtGui import *

from GUI import Ui_MainWindow  # 导入创建的GUI类

# leftEye = []
# rightEye = []

# 模块初始化
engine = pyttsx3.init()
content_1 = "眼睛闭上挺久了"
content_2 = "啊"

class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        self.count = 0
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.timer_calculator = QtCore.QTimer()  # 定义定时器，用来控制计算的频率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头

        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)
        self.button_close.clicked.connect(QCoreApplication.instance().quit)
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        # self.timer_calculator.timeout.connect(self.show_result)

    def eye_aspect_ratio(self, eye):  # 根据论文的公式，计算6个点，3对点的距离
        # 计算距离，竖直的
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        # 计算距离，水平的
        C = distance.euclidean(eye[0], eye[3])
        # ear值
        ear = (A + B) / (2.0 * C)
        return ear

    '''槽函数：启动摄像头和定时器'''
    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                # self.textedit1.setPlainText("请检查相机和电脑是否连接正确!")
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.timer_calculator.start(200)
                # self.textedit1.setPlainText("成功打开相机/视频，正在进行人脸，眼睛，嘴巴和鼻子的检测!")
                self.button_open_camera.setText('停止检测')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.timer_calculator.stop()
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('开始检测')
            # self.textedit1.setPlainText("成功关闭相机!")

    '''槽函数：人脸检测和特征点绘制'''
    def show_camera(self):
        import time
        time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        face_cascade = cv2.CascadeClassifier("fenleiqi//haarcascade_frontalface_default.xml")
        detector = dlib.get_frontal_face_detector()
        dlib_facelandmark = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (640, 480))
        res = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(show, 1.3, 5)

        # 人脸数
        faces = detector(show, 0)
        # 待会要写的字体
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 标 68 个点
        if len(faces) != 0:
            # 检测到人脸
            leftEye = []
            rightEye = []
            for i in range(len(faces)):
                # 取特征点坐标
                landmarks = np.matrix([[p.x, p.y] for p in predictor(show, faces[i]).parts()])
                # gray = cv2.cvtColor(QtGui.QImage(res, res.shape[1], res.shape[0], res.shape[1] * 3, QtGui.QImage.Format_RGB888), cv2.COLOR_BGR2GRAY)
                # face_landmarks = dlib_facelandmark(gray, faces)
                # 分别取两个眼睛区域
                # (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]  # 42-48
                # (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]  # 36-42
                for idx, point in enumerate(landmarks):
                    if (idx in range(36, 42)):
                        x = landmarks[idx][0, 0]
                        y = landmarks[idx][0, 1]
                        leftEye.append((x, y))
                    if(idx in range(42, 48)):
                        x = landmarks[idx][0, 0]
                        y = landmarks[idx][0, 1]
                        rightEye.append((x, y))
                    # 68 点的坐标
                    pos = (point[0, 0], point[0, 1])
                    # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
                    cv2.circle(res, pos, 2, color=(139, 0, 0))
                    # 利用 cv2.putText 写数字 1-68
                    if(self.checkBox_eyes.isChecked()):
                        if(idx in range(36,48)): cv2.putText(res, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
                    if(self.checkBox_mouse.isChecked()):
                        if(idx in range(49,68)): cv2.putText(res, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
                    # if(self.checkBox_mouse.checkState(2)):
                    # cv2.putText(res, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

                # 找左眼
                # for n in range(36, 42):  # 36~41 代表左眼
                #     x = landmarks.part(n).x
                #     y = landmarks.part(n).y
                #     leftEye.append((x, y))

                # 找右眼
                # for n in range(42, 48):  # 42~47代表右眼
                #     x = landmarks.part(n).x
                #     y = landmarks.part(n).y
                #     rightEye.append((x, y))
            cv2.putText(res, "faces: " + str(len(faces)), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            '''眨眼检测'''
            left_ear = self.eye_aspect_ratio(leftEye)
            right_ear = self.eye_aspect_ratio(rightEye)
            EAR = (left_ear + right_ear) / 2
            EAR = round(EAR, 2)
            if EAR < 0.26 :
                self.count += 1
                if(self.count >= 3) :
                    self.label_wink.setText('闭挺久了')
                    # 设置要播报的Unicode字符串
                    engine.say(content_1)
                    # 等待语音播报完毕
                    engine.runAndWait()
                else :
                    self.label_wink.setText('闭上了')
                    engine.say(content_2)
                    engine.runAndWait()
            else :
                self.label_wink.setText('睁开中')
                self.count = 0
            '''眯眼时长'''
            self.label_squint.setText(str(self.count*30)+'ms')


        else:
            # 没有检测到人脸
            cv2.putText(res, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

        image = QtGui.QImage(res, res.shape[1], res.shape[0], res.shape[1] * 3, QtGui.QImage.Format_RGB888)
        png1 = QtGui.QPixmap(image).scaled(self.label_show_camera.width(), self.label_show_camera.height())
        self.label_show_camera.setPixmap(png1)

        # opencv格式不能直接显示，需要用下面代码转换一下
        # show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        # showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        # self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        # 我们显示一个消息框,两个按钮:“是”和“不是”。第一个字符串出现在titlebar。
        # 第二个字符串消息对话框中显示的文本。第三个参数指定按钮的组合出现在对话框中。
        # 最后一个参数是默认按钮，这个是默认的按钮焦点。
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        # 处理返回值，如果单击Yes按钮,关闭小部件并终止应用程序。否则我们忽略关闭事件。
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()



    '''def show_result(self):
        left_ear = self.eye_aspect_ratio(leftEye)
        # right_ear = self.eye_aspect_ratio(rightEye)
        # EAR = (left_ear + right_ear) / 2
        EAR = left_ear
        EAR = round(EAR, 2)
        self.label_6.setText(str(EAR))'''



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())