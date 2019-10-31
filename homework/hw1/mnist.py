import matplotlib
import matplotlib.pyplot as plt
matplotlib.get_backend()
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import numpy as np

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel,\
 QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QGridLayout, QRadioButton, QLineEdit
from PyQt5 import QtWidgets, QtCore
import easygui
import numpy as np
import cv2


# Layout 
class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        grid = QGridLayout()
        grid.addWidget(self.deepLearning(), 0, 0, 2, 1)
        self.setLayout(grid)

        self.setWindowTitle("MNIST example")
        self.resize(300, 400)

    def deepLearning(self):

        groupBox = QGroupBox("5. MNIST example")

        bt0 = QPushButton("5.1 Show Train Images")
        bt0.setFixedSize(300,70)
        bt0.clicked.connect(showimages)

        bt1 = QPushButton("5.2 Show Hyperparameters")
        bt1.setFixedSize(300,70)
        #bt1.clicked.connect(colorConversion)

        bt2 = QPushButton("5.3 Train 1 Epoch")
        bt2.setFixedSize(300,70)
        #bt2.clicked.connect(imageFlipping)
        

        bt3 = QPushButton("5.4 Show Training Result")
        bt3.setFixedSize(300,70)
        #bt3.clicked.connect(blending)
        l0 = QLabel()
        l0.setText("Test Image Index:")
        line0 = QLineEdit()

        bt4 = QPushButton("5.5 Inference")
        bt4.setFixedSize(300,70)

        gbox = QGridLayout()
        gbox.addWidget(bt0,0,0,1,2)
        gbox.addWidget(bt1,1,0,1,2)
        gbox.addWidget(bt2,2,0,1,2)
        gbox.addWidget(bt3,3,0,1,2)
        gbox.addWidget(l0,4,0)
        gbox.addWidget(line0,4,1)
        gbox.addWidget(bt4,5,0,1,2)
        groupBox.setLayout(gbox)

        return groupBox


def showimages(checked):
	# 5.1 load MINST training dataset
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	w=10
	h=10
	columns = 5
	rows = 2
	fig=plt.figure(figsize=(8, 8))
	for i in range(1, columns*rows +1):
	    fig.add_subplot(rows, columns, i)
	    plt.imshow(x_train[i], cmap='gray_r')
	    plt.text(4,4,str(y_train[i]))
	    #plt.rcParams["font.family"] = "Times New Roman"
	    #plt.rcParams["font.size"] = "20"
	plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())
    sys.exit()
