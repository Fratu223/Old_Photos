import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import keras
import os
import cv2
import shutil

def image_preprocessing(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255
    image = np.expand_dims(image, 0)
    return image

def images_number(images_path):
    i = 0
    for image_name in os.listdir(images_path):
        if 'spate' in image_name:
            continue
        i += 1    
    return i

def prediction_result(model, image):
    prediction = model.predict(image)
    result = 0
    for _, pred in enumerate(prediction.round()):
        result += int(pred[0])
    return result

unclassified_images_path = "D:\\Imagini\\Nesortate"
color_images_path = "D:\\Imagini\\Color"
black_and_white_images_path = "D:\\Imagini\\Alb_Negru"
model_path = "D:\\Imagini\\Model\\old_photos_model.h5"
model = keras.models.load_model(model_path)
last_pred = 0

try:
    while True:
        if len(os.listdir(unclassified_images_path)) == 0:
            print('No image detected')
            continue
        else:
            print('Image detected')
            image_name = os.listdir(unclassified_images_path)[0]
            image_path = unclassified_images_path + "\\" + image_name
            color_images_number = images_number(color_images_path)
            black_and_white_images_number = images_number(black_and_white_images_path)
            if image_name == 'spate.jpg':
                print('Back of image detected')
                if last_pred == 0:
                    shutil.move(image_path, black_and_white_images_path + '\\' + 'image_' + str(black_and_white_images_number) + '_alb_negru' + '_spate' + '.jpg')
                    print('Image moved succesfully')
                elif last_pred == 1:
                    shutil.move(image_path, color_images_path + '\\' + 'image_' + str(color_images_number) + '_color' + '_spate' + '.jpg')
                    print('Image moved succesfully')
            else:
                print('Front of image detected')
                image = image_preprocessing(image_path)
                pred = prediction_result(model, image)
                if pred == 0:
                    shutil.move(image_path, black_and_white_images_path + '\\' + 'image_' + str(black_and_white_images_number + 1) + '_alb_negru' + '.jpg')
                    print('Image moved succesfully')
                elif pred == 1:
                    shutil.move(image_path, color_images_path + '\\' + 'image_' + str(color_images_number + 1) + '_color' + '.jpg')
                    print('Image moved succesfully')
                last_pred = pred

except KeyboardInterrupt:
    pass