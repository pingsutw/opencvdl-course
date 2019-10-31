import matplotlib
import matplotlib.pyplot as plt
matplotlib.get_backend()

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel,\
 QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QGridLayout, QRadioButton, QLineEdit
from PyQt5 import QtWidgets, QtCore
import easygui
import numpy as np
import cv2

image_name = "images/QR.png"

# Layout 
class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        grid = QGridLayout()
        grid.addWidget(self.imageProcessing(), 0, 0, 2, 1)
        grid.addWidget(self.adaptiveThreshold(), 0, 1)
        grid.addWidget(self.convolution(), 1, 1)
        grid.addWidget(self.imageTransformation(), 0, 2, 2, 1)
        self.setLayout(grid)

        self.setWindowTitle("PyQt5 Group Box")
        self.resize(800, 400)

    def imageProcessing(self):

        groupBox = QGroupBox("1. ImageProcessing")

        bt0 = QPushButton("1.1 Load Imgae")
        bt0.setFixedSize(200,60)
        bt0.clicked.connect(loadImage)

        bt1 = QPushButton("1.2 Color Conversion")
        bt1.setFixedSize(200,60)
        bt1.clicked.connect(colorConversion)

        bt2 = QPushButton("1.3 Image Flipping")
        bt2.clicked.connect(imageFlipping)
        bt2.setFixedSize(200,60)

        bt3 = QPushButton("1.4 Blending")
        bt3.setFixedSize(200,60)
        bt3.clicked.connect(blending)

        vbox = QVBoxLayout()
        vbox.addWidget(bt0)
        vbox.addWidget(bt1)
        vbox.addWidget(bt2)
        vbox.addWidget(bt3)
        vbox.addStretch(4)
        groupBox.setLayout(vbox)
        #groupBox.setFixedSize(220,400)

        return groupBox

    def adaptiveThreshold(self):
        groupBox = QGroupBox("2. Adaptive Threshold")

        bt0 = QPushButton("2.1 Global Threshold")
        bt0.setFixedSize(200,60)
        bt0.clicked.connect(globalThreshold)

        bt1 = QPushButton("2.2 Local Threshold")
        bt1.setFixedSize(200,60)
        bt1.clicked.connect(localThreshold)

        vbox = QVBoxLayout()
        vbox.addWidget(bt0,0)
        vbox.addWidget(bt1,1)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    def imageTransformation(self):
        groupBox = QGroupBox("3. Image Transformation")

        box = QGroupBox("3.1 Rot, scale, Translate")
        box.setFixedSize(300,300)

        box_child = QGroupBox("Parameters")
        box_child.setFixedSize(280,190)
        grid_child = QGridLayout()
        l0 = QLabel()
        l0.setText("Angle:")
        l1 = QLabel()
        l1.setText("Scale:")
        l2 = QLabel()
        l2.setText("Tx")
        l3 = QLabel()
        l3.setText("Ty:")

        grid_child.addWidget(l0,0,0)
        grid_child.addWidget(l1,1,0)
        grid_child.addWidget(l2,2,0)
        grid_child.addWidget(l3,3,0)
        box_child.setLayout(grid_child)

        line0 = QLineEdit()
        line1 = QLineEdit()
        line2 = QLineEdit()
        line3 = QLineEdit()
        grid_child.addWidget(line0,0,1)
        grid_child.addWidget(line1,1,1)
        grid_child.addWidget(line2,2,1)
        grid_child.addWidget(line3,3,1)

        l4 = QLabel()
        l4.setText("deg")
        l5 = QLabel()
        l5.setText("pixel")
        l6 = QLabel()
        l6.setText("pixel")

        grid_child.addWidget(l4,0,2)
        grid_child.addWidget(l5,2,2)
        grid_child.addWidget(l6,3,2)
        box_child.setLayout(grid_child)

        bt0 = QPushButton("3.1 Rotation, scaling, translation")
        bt0.setFixedSize(280,60)

        vbox_child = QVBoxLayout()
        vbox_child.addWidget(box_child)
        vbox_child.addWidget(bt0)

        box.setLayout(vbox_child)

        bt1 = QPushButton("3.2 persperctive Transform")
        bt1.setFixedSize(300,60)

        vbox = QVBoxLayout()
        vbox.addWidget(box,0)
        vbox.addWidget(bt1,1)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox


    def convolution(self):
        groupBox = QGroupBox("4. Convolution")

        bt0 = QPushButton("4.1 Gaussian")
        bt0.setFixedSize(200,60)
        bt1 = QPushButton("4.2 Sobel X")
        bt1.setFixedSize(200,60)
        bt2 = QPushButton("4.3 Sobel Y")
        bt2.setFixedSize(200,60)
        bt3 = QPushButton("4.4 Magnitude")
        bt3.setFixedSize(200,60)

        vbox = QVBoxLayout()
        vbox.addWidget(bt0,0)
        vbox.addWidget(bt1,1)
        vbox.addWidget(bt2,2)
        vbox.addWidget(bt3,3)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox


# Event
def showImage(name, img):
	plt.imshow(img)
	plt.show()
	# cv2.imshow(name, img)
	# cv2.waitKey(0)
	# cv2.destroyWindow(name)


def loadImage(checked):
	global image_name
	image_name = easygui.fileopenbox()
	img = cv2.imread(image_name)
	print("Height", img.shape[0])
	print("Width", img.shape[1])
	showImage("loadImage", img)

def colorConversion(checked):
	img = cv2.imread(image_name)
	im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	showImage("colorConversion", im_rgb)

def imageFlipping(checked):
	img = cv2.imread(image_name)
	flipVertical = cv2.flip(img, 1)
	showImage("imageFlipping", flipVertical)

def nothing(x):
    pass

def blending(checked):
    img = cv2.imread(image_name)
    flipVertical = cv2.flip(img, 1)
    alpha = 0.5
    cv2.namedWindow('blending')
    cv2.createTrackbar('alpla','blending',0,255,nothing)
    cv2.setTrackbarPos('alpla', 'blending', 255//2)
    blend = cv2.addWeighted(img, alpha, flipVertical, 1-alpha, 0.0)
    while(1):
        cv2.imshow('blending', blend)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        alpha = cv2.getTrackbarPos('alpla','blending')
        if alpha > 0:
            alpha /= 255  
        blend = cv2.addWeighted(img, alpha, flipVertical, 1-alpha, 0.0)
    cv2.destroyWindow('blending') 

def globalThreshold(checked):
    img = cv2.imread(image_name)
    ret, thresh = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
    showImage("globalThreshold", thresh)

def localThreshold(checked):
	img = cv2.imread(image_name, cv2.CV_8UC1)
	thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,19,-1)
	showImage("localThreshold", thresh)

def imageTransformation(checked):
	img = cv2.imread(image_name)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())
    sys.exit()