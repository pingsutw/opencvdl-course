import sys

from PyQt5 import uic
from PyQt5.QtCore import Qt, QTime
from PyQt5.QtWidgets import QApplication, QMainWindow

import cv2
import numpy as np

from gui.windows import ImageWindow, SlideWindow, MultiImageWindow


class MainWindow(QMainWindow):

    widgets = []

    def __init__(self):
        super().__init__()
        uic.loadUi('D:/1.ui', self)

        self.bindUi()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            for w in self.widgets:
                w.close()

            self.close()

    def addWidget(self, w):
        self.widgets.append(w)

    def bindUi(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        pass

    def on_btn1_1_click(self):
        imageL = cv2.imread('D:/data/imL.png', cv2.IMREAD_GRAYSCALE)
        imageR = cv2.imread('D:/data/imR.png', cv2.IMREAD_GRAYSCALE)

        stereo = cv2.StereoBM_create(64, 9)
        disparity = stereo.compute(imageL, imageR)
        image = np.zeros_like(disparity)

        cv2.normalize(disparity, image, 0, 255, cv2.NORM_MINMAX)

        w = ImageWindow(title='1.1 Disparity', image=image)
        w.show()

        self.addWidget(w)

    def on_btn2_1_click(self):
        video = cv2.VideoCapture('D:/data/bgSub.mp4')

        if not video.isOpened():
            print('video not found')
            return

        w = None
        subtractor = cv2.createBackgroundSubtractorMOG2()#detectShadows=False)
        t = QTime()

        t.start()

        while video.isOpened():
            ret, frame = video.read()

            if ret:
                fg = subtractor.apply(frame)

                if w is None:
                    w = MultiImageWindow(
                        title='2.1 Background Subtraction', images=[frame, fg])
                    w.show()

                    self.addWidget(w)
                else:
                    w.setImage(frame, 0)
                    w.setImage(fg, 1)
                    w.update()

                t.restart()

                while t.elapsed() < 33:
                    QApplication.processEvents()
            else:
                break

        video.release()

    def on_btn3_1_click(self):
        video = cv2.VideoCapture('D:/data/featureTracking.mp4')

        detector = self._get_blob_detector()

        if not video.isOpened():
            print('video not found')
            return

        w = None

        while video.isOpened():
            ret, frame = video.read()

            if ret:
                keypoints = detector.detect(frame)

                for kp in keypoints:
                    p = tuple(int(round(c)) for c in kp.pt)
                    p1 = tuple(int(round(c - 5.5)) for c in kp.pt)
                    p2 = tuple(int(round(c + 5.5)) for c in kp.pt)

                    frame = cv2.rectangle(frame, p1, p2, (0, 0, 255), 1)
                    frame = cv2.line(
                        frame,
                        (p1[0], p[1]),
                        (p2[0], p[1]),
                        (0, 0, 255),
                        1,
                    )
                    frame = cv2.line(
                        frame,
                        (p[0], p1[1]),
                        (p[0], p2[1]),
                        (0, 0, 255),
                        1,
                    )

                if w is None:
                    w = ImageWindow(title='3.1 Preprocessing', image=frame)
                    w.show()

                    self.addWidget(w)
                else:
                    w.setImage(frame)
                    w.update()

                QApplication.processEvents()
            else:
                break

            break

        video.release()

    def on_btn3_2_click(self):
        video = cv2.VideoCapture('D:/data/featureTracking.mp4')

        if not video.isOpened():
            print('video not found')
            return

        w = None
        prev_img = None
        tracks = []
        t = QTime()

        t.start()

        while video.isOpened():
            ret, frame = video.read()

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
                    detector = self._get_blob_detector()
                    keypoints = detector.detect(frame)

                    pts = [tuple(kp.pt) for kp in keypoints]

                    prev_img = gray
                    tracks = [[p] for p in pts]

                if w is None:
                    w = ImageWindow(title='3.2 Video tracking', image=frame)
                    w.show()

                    self.addWidget(w)
                else:
                    w.setImage(frame)
                    w.update()

                t.restart()

                while t.elapsed() < 33:
                    QApplication.processEvents()
            else:
                break

        video.release()

    def on_btn4_1_click(self):
        images = [cv2.imread(f'D:/data/{i+1}.bmp', cv2.IMREAD_GRAYSCALE)
                  for i in range(5)]

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

            image = cv2.line(image, cs[0], cs[1], [0, 0, 255], 10)
            image = cv2.line(image, cs[0], cs[2], [0, 0, 255], 10)
            image = cv2.line(image, cs[0], cs[3], [0, 0, 255], 10)
            image = cv2.line(image, cs[0], cs[4], [0, 0, 255], 10)

            image = cv2.line(image, cs[1], cs[2], [0, 0, 255], 10)
            image = cv2.line(image, cs[2], cs[3], [0, 0, 255], 10)
            image = cv2.line(image, cs[3], cs[4], [0, 0, 255], 10)
            image = cv2.line(image, cs[4], cs[1], [0, 0, 255], 10)

            image = cv2.resize(image, (512, 512))

            return image

        images = [draw_pyramid(image, r[:, i, np.newaxis], t[:, i, np.newaxis])
                  for i, image in enumerate(images)]

        w = SlideWindow(images, title='4.1 Augmented Reality')
        w.show()

        self.addWidget(w)

    def _get_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.83
        params.filterByArea = True
        params.minArea = 30.
        params.maxArea = 100.

        return cv2.SimpleBlobDetector_create(params)


if _name_ == '_main_':
    a = QApplication(sys.argv)
    w = MainWindow()

    w.show()

    sys.exit(a.exec_())