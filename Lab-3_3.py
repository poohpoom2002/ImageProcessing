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

model = VGG16(weights='imagenet', include_top=False)

conv_layer = model.layers[1]
kernels = conv_layer.get_weights()[0]

img_result = np.zeros((224, 224, 3))
image_sum = np.zeros((224, 224, 64))

for i in range(64):
    for channel in range(3):
        img_result[:, :, channel] = signal.convolve2d(img_mean_subtracted[0, :, :, channel], kernels[:, :, channel, i], mode='same', boundary='fill', fillvalue=0)
    image_sum[:, :, i] = img_result[:, :, 0] + img_result[:, :, 1] + img_result[:, :, 2]

image_sum[image_sum < 0] = 0

plt.figure(figsize=(8, 8))
for i in range(64):  # Display the first 64 channels
    plt.subplot(8, 8, i+1)
    plt.imshow(image_sum[:, :, i], cmap='gray')
    plt.axis('off')
plt.show()