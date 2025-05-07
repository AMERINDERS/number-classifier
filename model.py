#import cv2 as cv
import numpy as np
#import matplot.pyplot as plt#
import tensorflow as tf

mnist=tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test)=mnist.load_data()

x_train= tf.keras.utils.normalize(x_train,axis=1)
x_test= tf.keras.utils.normalize(x_test,axis=1)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),                           
    tf.keras.layers.Dense(128, activation='relu'),       
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)

accuracy,loss=model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

model.save('digits.model.keras')