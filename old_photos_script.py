import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import keras
import os
import cv2

unclassified_images_path = "D:\\Imagini\\Nesortate"
image_name = os.listdir(unclassified_images_path)[0]
image_path = unclassified_images_path + "\\" + image_name
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = image / 255
image = np.expand_dims(image, 0)
model_path = "D:\\Imagini\\Model\\old_photos_model.h5"
model = keras.models.load_model(model_path)
prediction = model.predict(image)
print(int(prediction[0]))