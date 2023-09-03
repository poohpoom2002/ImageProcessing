import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from scipy import signal

img = cv2.imread("Pee-Saderd.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img_4D = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

img_mean = [123.68, 116.779, 103.939]

img_mean_subtracted = img_4D - img_mean
img_mean_subtracted = img_mean_subtracted[:, :, :, ::-1]

img_array = img_to_array(img)
img_array = expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(img_array, data_format='channels_last')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Preprocessed Image by Function')
plt.imshow(preprocessed_img[0])
plt.subplot(1, 2, 2)
plt.title('Preprocessed Image by Scratch')
plt.imshow(img_mean_subtracted[0])
plt.show()