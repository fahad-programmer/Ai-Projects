import tensorflow as tf


#Data set that we will be using through
mnist = tf.keras.datasets.mnist

#Splitting the data in testing and traning 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizing the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Creating the mode 
"""
Following Layers are there

1 Input Layer
2 Dense Layer
1 Output Layer

"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save("digits.model")



import cv2 as Cv
import numpy as np
import matplotlib.pyplot as plt


try:
    img = Cv.imread("1.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"The Result Is Probably: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
except Exception as e:
    print("An error occurred:", e)
