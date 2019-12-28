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
    params.minThreshold = 30
    params.maxThreshold = 100
    params.filterByArea = True
    params.filterByCircularity = True
    params.minCircularity = 0.83
    params.minArea = 10
    detector = cv2.SimpleBlobDetector_create(params)

    cap = cv2.VideoCapture('images/featureTracking.mp4')
    feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

    lk_params = dict(winSize=(21, 21),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))

    ret, old_frame = cap.read()
    # keypoints = detector.detect(old_frame)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('test1',img)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()


def ar(checked):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('images/*.bmp')

    for fname in images:
        print("read image")
        img = cv2.imread(fname)
        img = cv2.resize(img,(240,240))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,11),None)
        # cv2.imshow('img',img)
        # cv2.waitKey(500)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("Find object points")
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (8,11), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(1000)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
    axis = np.float32( [ [0,0,0], [0,6,0], [6,6,0], [6,0,0], [0,0,-3],[0,6,-3],[6,6,-3],[6,0,-3] ] )
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    cv2.imshow('img',imgpts)
    cv2.waitKey(1000)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
    sys.exit()