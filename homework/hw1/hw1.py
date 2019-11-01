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

image_name = "images/School.jpg"
# Transformation attribute
angle = None
Scale = None
Tx = None
Ty = None


########################## Layout ################################
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

        label0 = QLabel()
        label0.setText("Angle:")
        label1 = QLabel()
        label1.setText("Scale:")
        label2 = QLabel()
        label2.setText("Tx")
        label3 = QLabel()
        label3.setText("Ty:")

        grid_child.addWidget(label0,0,0)
        grid_child.addWidget(label1,1,0)
        grid_child.addWidget(label2,2,0)
        grid_child.addWidget(label3,3,0)
        box_child.setLayout(grid_child)

        line0 = QLineEdit()
        line0.setText("45")
        line1 = QLineEdit("0.8")
        line2 = QLineEdit("150")
        line3 = QLineEdit("50")

        grid_child.addWidget(line0,0,1)
        grid_child.addWidget(line1,1,1)
        grid_child.addWidget(line2,2,1)
        grid_child.addWidget(line3,3,1)

        label4 = QLabel()
        label4.setText("deg")
        label5 = QLabel()
        label5.setText("pixel")
        label6 = QLabel()
        label6.setText("pixel")

        grid_child.addWidget(label4,0,2)
        grid_child.addWidget(label5,2,2)
        grid_child.addWidget(label6,3,2)
        box_child.setLayout(grid_child)

        def button_click(self):
            global angle, Scale, Tx, Ty
            angle = float(line0.text())
            Scale = float(line1.text())
            Tx = float(line2.text())
            Ty = float(line3.text())
            imageTransformation()

        bt0 = QPushButton("3.1 Rotation, scaling, translation")
        bt0.setFixedSize(280,60)
        bt0.clicked.connect(button_click)
        
        vbox_child = QVBoxLayout()
        vbox_child.addWidget(box_child)
        vbox_child.addWidget(bt0)

        box.setLayout(vbox_child)

        bt1 = QPushButton("3.2 persperctive Transform")
        bt1.setFixedSize(300,60)
        bt1.clicked.connect(perspectiveTransformation)

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
        bt0.clicked.connect(gaussian)

        bt1 = QPushButton("4.2 Sobel X")
        bt1.setFixedSize(200,60)
        bt1.clicked.connect(sobelX)

        bt2 = QPushButton("4.3 Sobel Y")
        bt2.setFixedSize(200,60)
        bt2.clicked.connect(sobelY)

        bt3 = QPushButton("4.4 Magnitude")
        bt3.setFixedSize(200,60)
        bt3.clicked.connect(magnitude)

        vbox = QVBoxLayout()
        vbox.addWidget(bt0,0)
        vbox.addWidget(bt1,1)
        vbox.addWidget(bt2,2)
        vbox.addWidget(bt3,3)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

########################## Event ################################
points = []

def showImage(name, img):
    fig, ax = plt.subplots()

    def onclick(event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))
        circle=plt.Circle((event.xdata,event.ydata),12,color='red')
        ax.add_patch(circle)
        fig.canvas.draw()
        global points
        points.append([event.xdata, event.ydata])
        if len(points) == 4:
            showPerspective(points)
            points = []

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.set_window_title(name)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def cvShowImage(name, img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyWindow(name)

def loadImage(checked):
	global image_name
	image_name = easygui.fileopenbox()
	img = cv2.imread(image_name)
	print("Height", img.shape[0])
	print("Width", img.shape[1])
	showImage("loadImage", img)

def colorConversion(checked):
    img = cv2.imread("images/color.png")
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = img[...,::-1].copy()
    gbr = rgb[...,[2,0,1]].copy()
    rbg = gbr[...,::-1].copy()
    showImage("colorConversion", rbg)

def imageFlipping(checked):
	img = cv2.imread("images/dog.bmp")
	flipVertical = cv2.flip(img, 1)
	showImage("imageFlipping", flipVertical)

def nothing(x):
    pass

def blending(checked):
    img = cv2.imread("images/dog.bmp")
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
    img = cv2.imread('images/QR.png')
    ret, thresh = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
    showImage("globalThreshold", thresh)

def localThreshold(checked):
    img = cv2.imread('images/QR.png', cv2.CV_8UC1)
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,19,-1)
    showImage("localThreshold", thresh)

def imageTransformation():
    img = cv2.imread('images/OriginalTransform.png')
    (h, w) = img.shape[:2]
    center = (130, 125)

    rotated = cv2.getRotationMatrix2D(center, angle, Scale)
    transform = np.float32([[1,0,Tx],[0,1,Ty]])

    final = cv2.warpAffine(img, rotated, (w, h))
    final = cv2.warpAffine(final,transform,(w, h))
    
    showImage("imageTransformation", final)


def perspectiveTransformation():
    img = cv2.imread('images/OriginalPerspective.png')
    showImage("OriginalPerspective", img)

def showPerspective(points):
	img = cv2.imread('images/OriginalPerspective.png')
	pts1 = np.float32([points[0], points[1], points[2], points[3]])
	pts2 = np.float32([[20, 20], [450, 20], [450, 450], [20, 450]])
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	perspective = cv2.warpPerspective(img, matrix, (430, 430))
	showImage("perspectiveTransformation", perspective)

def convolve2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))    
    output = np.zeros_like(image)            
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     
        for y in range(image.shape[0]):
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
    return output

def gaussian(checked):
    img = cv2.imread('images/School.jpg')
    img = np.dot(img[...,:3], [0.1140, 0.5870, 0.2989])

    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.title('Grayscale')
    plt.imshow(img, cmap = plt.get_cmap(name = 'gray'))

    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaus = convolve2d(img, gaussian_kernel)

    plt.subplot(2,1,2)
    plt.title('Gaussian smoothing')
    plt.imshow(gaus, cmap = plt.get_cmap(name = 'gray'))
    plt.show()

def sobelX(checked):
    img = cv2.imread('images/School.jpg')
    img = np.dot(img[...,:3], [0.1140, 0.5870, 0.2989])

    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    img = convolve2d(img, gaussian_kernel)

    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.title('Gaussian smoothing')
    plt.imshow(img, cmap = plt.get_cmap(name = 'gray'))

    filterx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img = convolve2d(img, filterx)
    
    plt.subplot(2,1,2)
    plt.title('SobelX')
    plt.imshow(img, cmap = plt.get_cmap(name = 'gray'))
    plt.show()

def sobelY(checked):
    img = cv2.imread('images/School.jpg')
    img = np.dot(img[...,:3], [0.1140, 0.5870, 0.2989])

    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    img = convolve2d(img, gaussian_kernel)

    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.title('Gaussian smoothing')
    plt.imshow(img, cmap = plt.get_cmap(name = 'gray'))

    filtery = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    img = convolve2d(img, filtery)
    
    plt.subplot(2,1,2)
    plt.title('SobelY')
    plt.imshow(img, cmap = plt.get_cmap(name = 'gray'))
    plt.show()


def magnitude(checked):
    img = cv2.imread('images/School.jpg')
    img = np.dot(img[...,:3], [0.1140, 0.5870, 0.2989])

    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    img = convolve2d(img, gaussian_kernel)

    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.title('Gaussian smoothing')
    plt.imshow(img, cmap = plt.get_cmap(name = 'gray'))

    filterx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    imgx = convolve2d(img, filterx)

    filtery = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    imgy = convolve2d(img, filtery)

    final = np.sqrt(np.square(imgx) + np.square(imgy))

    plt.subplot(2,1,2)
    plt.title('magnitude')
    plt.imshow(final, cmap = plt.get_cmap(name = 'gray'))
    plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
    sys.exit()