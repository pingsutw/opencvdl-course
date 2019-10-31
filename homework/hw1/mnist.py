import matplotlib
import matplotlib.pyplot as plt
matplotlib.get_backend()
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 5.1 load MINST training dataset
w=10
h=10
columns = 5
rows = 2
fig=plt.figure(figsize=(8, 8))
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(x_train[i], cmap='gray_r')
    plt.text(4,4,str(y_train[i]))
    plt.rcParams["font.family"] = "Times New Roman"
    #plt.rcParams["font.size"] = "20"
plt.show()
