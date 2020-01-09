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
import glob

########################## Layout ################################
class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        grid = QGridLayout()
        grid.addWidget(self.stereo(), 0, 0, 2, 1)
        grid.addWidget(self.backgroundSubstraction(), 0, 1)
        grid.addWidget(self.featureTracking(), 1, 1)
        grid.addWidget(self.argumentedReality(), 0, 2, 2, 1)
        self.setLayout(grid)

        self.setWindowTitle("PyQt5 Group Box")
        self.resize(800, 400)

    def stereo(self):

        groupBox = QGroupBox("1. Stereo")

        bt0 = QPushButton("1.1 Disparity")
        bt0.setFixedSize(200,60)
        bt0.clicked.connect(Disparity)

        vbox = QVBoxLayout()
        vbox.addWidget(bt0)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    def backgroundSubstraction(self):
        groupBox = QGroupBox("2. Background substraction")

        bt0 = QPushButton("2.1 Background substraction")
        bt0.setFixedSize(200,60)
        bt0.clicked.connect(subtraction)

        vbox = QVBoxLayout()
        vbox.addWidget(bt0,0)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    def featureTracking(self):
        groupBox = QGroupBox("3. Feature Tracking")

        bt0 = QPushButton("3.1 preprocssing")
        bt0.setFixedSize(200,60)
        bt0.clicked.connect(preprocessing)

        bt1 = QPushButton("3.2 Video tracking")
        bt1.setFixedSize(200,60)
        bt1.clicked.connect(tracking)

        vbox = QVBoxLayout()
        vbox.addWidget(bt0,0)
        vbox.addWidget(bt1,1)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    def argumentedReality(self):
        groupBox = QGroupBox("4. argumentedReality")

        bt0 = QPushButton("4.1 argumentedReality")
        bt0.setFixedSize(200,60)
        bt0.clicked.connect(ar)

        vbox = QVBoxLayout()
        vbox.addWidget(bt0,0)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

########################## Event ################################
def Disparity(checked):
    imgL = cv2.imread('images/imL.png',0)
    imgR = cv2.imread('images/imR.png',0)
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()

def subtraction(checked):
    cap = cv2.VideoCapture('images/bgSub.mp4')
    backSub = cv2.createBackgroundSubtractorMOG2()
    while(cap.isOpened()):
        ret, frame = cap.read()
        fgMask = backSub.apply(frame)

        cv2.imshow('test1',frame)
        cv2.imshow('test2',fgMask)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def preprocessing(checked):
    cap = cv2.VideoCapture('images/featureTracking.mp4')
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 30
    params.maxThreshold = 100
    params.filterByArea = True
    params.filterByCircularity = True
    params.minCircularity = 0.83
    params.minArea = 10

    detector = cv2.SimpleBlobDetector_create(params)
    while(cap.isOpened()):
        ret, frame = cap.read()
        keypoints = detector.detect(frame)
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('test1',im_with_keypoints)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def tracking(checked):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 30.
    params.maxThreshold = 100.
    params.filterByArea = True
    params.filterByCircularity = True
    params.minCircularity = 0.83
    # params.minArea = 10
    detector = cv2.SimpleBlobDetector_create(params)

    cap = cv2.VideoCapture('images/featureTracking.mp4')
    tracks = []
    prev_img = None
    
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_img is not None:
                prev_pts = np.array([tr[-1]
                                         for tr in tracks], dtype=np.float32).reshape(-1, 1, 2)

                p1, _, _ = cv2.calcOpticalFlowPyrLK(
                    prev_img, gray, prev_pts, None, None, winSize=(21, 21))
                p0, _, _ = cv2.calcOpticalFlowPyrLK(
                    gray, prev_img, p1, None, None, winSize=(21, 21))
                d = abs(prev_pts - p0).max(-1)

                prev_img = gray

                pts = [kp.pt for kp in keypoints]
                pts = [tuple([int(round(c)) for c in p]) for p in pts]

                for tr, (x, y), valid in zip(tracks, p1.reshape(-1, 2), d < 50):
                    if not valid:
                        continue

                    tr.append((x, y))

                    x, y = int(round(x)), int(round(y))

                frame = cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)
                frame = cv2.polylines(
                frame, [np.int32(tr) for tr in tracks], False, (0, 0, 255), 2)

            else:
                keypoints = detector.detect(frame)
                pts = [tuple(kp.pt) for kp in keypoints]
                prev_img = gray
                tracks = [[p] for p in pts]

            cv2.imshow('tracking',frame)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def ar(checked):
    intrinsic = np.array([
            [2.22549585e+03, 0.00000000e+00, 1.02554596e+03],
            [0.00000000e+00, 2.22518414e+03, 1.03858519e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ], dtype=np.float32)

    distortion = np.array([[
        -1.28742249e-01,
        9.05778153e-02,
        -9.91253306e-04,
        2.77634515e-06,
        2.29249706e-03,
    ]], dtype=np.float32)

    r = np.array([
        [0.36242004, -0.69610376, -1.46189723, -1.200479, 0.65333595],
        [0.35199599, 0.24413134, -0.35665533, -1.17458499, -0.10943514],
        [3.05459015, 2.93842086, -2.61331073, -2.35813427, 2.84431054],
    ], dtype=np.float32)

    t = np.array([
        [6.81253889, 3.3925504,  2.68774801, 1.22781875, 4.43641198],
        [3.37330384, 4.36149229, 4.70990021, 3.48023006, 0.67177428],
        [16.71572319, 22.15957429, 12.98147662, 10.9840538, 16.24069227],
    ], dtype=np.float32)


    def draw_pyramid(image, r, t):
            corners = np.array([
                (3, 3, -4),
                (5, 5,  0),
                (1, 5,  0),
                (1, 1,  0),
                (5, 1,  0),
            ], dtype=np.float32)

            cs, _ = cv2.projectPoints(corners, r, t, intrinsic, distortion)
            cs = np.squeeze(cs, axis=1)
            cs = [tuple(c) for c in cs]

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            for i in range (1,5):
                image = cv2.line(image, cs[0], cs[i], [0, 0, 255], 10)

            image = cv2.line(image, cs[1], cs[2], [0, 0, 255], 10)
            image = cv2.line(image, cs[2], cs[3], [0, 0, 255], 10)
            image = cv2.line(image, cs[3], cs[4], [0, 0, 255], 10)
            image = cv2.line(image, cs[4], cs[1], [0, 0, 255], 10)

            image = cv2.resize(image, (512, 512))

            return image

    images = [cv2.imread(f'./images/{i+1}.bmp', cv2.IMREAD_GRAYSCALE)
                  for i in range(5)]
    images = [draw_pyramid(image, r[:, i, np.newaxis], t[:, i, np.newaxis])
                  for i, image in enumerate(images)]

    for image in images:
        cv2.imshow('img',image)
        cv2.waitKey(1000)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
    sys.exit()