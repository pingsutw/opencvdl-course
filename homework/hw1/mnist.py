import numpy as np
import sys
# matplotlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.get_backend()

# pyqt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel,\
 QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QGridLayout, QRadioButton, QLineEdit
import easygui

# OpenCV
import cv2

# Tensorflow 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
mnist = tf.keras.datasets.mnist

index = None

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
        bt1.clicked.connect(hyperparameters)

        bt2 = QPushButton("5.3 Train 1 Epoch")
        bt2.setFixedSize(300,70)
        bt2.clicked.connect(trainOneEpoch)
        

        bt3 = QPushButton("5.4 Show Training Result")
        bt3.setFixedSize(300,70)
        bt3.clicked.connect(trainingResult)
        label0 = QLabel()
        label0.setText("Test Image Index:")
        line0 = QLineEdit()
        line0.setText("0")

        def button_click(self):
            global index
            index = int(line0.text())
            inference()

        bt4 = QPushButton("5.5 Inference")
        bt4.setFixedSize(300,70)
        bt4.clicked.connect(button_click)

        gbox = QGridLayout()
        gbox.addWidget(bt0,0,0,1,2)
        gbox.addWidget(bt1,1,0,1,2)
        gbox.addWidget(bt2,2,0,1,2)
        gbox.addWidget(bt3,3,0,1,2)
        gbox.addWidget(label0,4,0)
        gbox.addWidget(line0,4,1)
        gbox.addWidget(bt4,5,0,1,2)
        groupBox.setLayout(gbox)

        return groupBox


def showimages(checked):
	# 5.1 load MINST training dataset
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# image 28 * 28 
	print("x_train.shape :", x_train.shape)
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

n_classes = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
OPTIMIZER = "SGD"

def hyperparameters(checked):
	print("hyperparameters:")
	print("batch size:", BATCH_SIZE)
	print("leraning rate:", LEARNING_RATE)
	print("optimizer:", OPTIMIZER)


def getModel():
    inputs = keras.Input(shape=(28, 28, 1), name='img')
    x = layers.Conv2D(6, 5, 1, activation='tanh')(inputs)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(16, 5, 1, activation='tanh')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(84, activation='tanh')(x)
    x = layers.Dense(10, activation='tanh')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='Lenet')
    print(model.summary())
    return model
    

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def trainOneEpoch(checked):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # x_train = x_train.reshape(60000, 784).astype('float32') / 255
    # x_test = x_test.reshape(10000, 784).astype('float32') / 255
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape(-1,28, 28, 1)
    x_test = x_test.reshape(-1,28, 28, 1)

    print(x_train[0].shape)
    model = getModel()
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

    lossHistory = LossHistory()
    history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=1,
                    validation_split=0.2,
                    callbacks=[lossHistory])
    # print(lossHistory.losses)
    print(history.history.keys())
    plt.plot(lossHistory.losses)
    plt.title('epoch [0/50]')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend(['train'], loc='upper left')
    plt.show()
   
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])
    model.save('model/mnist-epoch1.h5')

def trainingResult():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape(-1,28, 28, 1)
    x_test = x_test.reshape(-1,28, 28, 1)

    model = getModel()
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=50,
                    validation_split=0.2)

    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('%')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Testing'], loc='lower right')
    
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Testing'], loc='lower right')
    plt.show()

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])
    model.save('model/mnist.h5')

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


def inference():
    (x_train, y_train), (origin_x_test, y_test) = mnist.load_data()
    x_test = origin_x_test.astype('float32') / 255.
    x_test = x_test.reshape(-1,28, 28, 1)

    model = tf.keras.models.load_model('model/mnist.h5')

    predictions = model.predict(x_test)
    print(predictions[index].shape)

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(index, predictions[index], y_test, origin_x_test)
    plt.subplot(1,2,2)
    plot_value_array(index, predictions[index],  y_test)
    plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
    sys.exit()
