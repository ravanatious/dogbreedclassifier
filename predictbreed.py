from tensorflow import keras
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import decode_predictions, preprocess_input, InceptionV3
import cv2
import csv

file = open('dogclassification_breed.csv')
csvreader = csv.reader(file)

breeds = []
for row in csvreader:
    breeds.append(row)
file.close()

print(breeds)

model = keras.models.load_model('static/inceptionv3_rev2')



img = cv2.imread('static/golden retriever.jpg')
img_resized = cv2.resize(img, (299, 299))
img_preprocessed = preprocess_input(img_resized)
img_reshaped = img_preprocessed.reshape((1, 299, 299, 3))
prediction = model.predict(img_reshaped)

index = np.argmax(prediction)
value = prediction[0][index] * 100



print(f'This is a: {breeds[index]} with a {value}  accuracy')
