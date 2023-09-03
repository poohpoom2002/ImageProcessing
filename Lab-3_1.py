import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from scipy import signal

model = VGG16(weights='imagenet', include_top=False)

image = cv2.imread("Pee-Saderd.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = cv2.resize(image, (224, 224))
# convert the image to an array
img_array = img_to_array(img)
# expand dimensions so that it represents a single 'sample’
# reshape 3D(H,W,Ch) image to 4D image (sample=1,H,W,Ch)
img_array = expand_dims(img_array, axis=0)

# prepare the image (e.g. scale pixel values for the vgg)
preprocessed_img = preprocess_input(img_array)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title('Preprocessed Image')
plt.imshow(preprocessed_img[0])
plt.show()

# Fetch CNN Layer 1 specific VGG16 model
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()

# Extract feature maps from CNN Layer 1
feature_maps = model.predict(preprocessed_img)

# Display the first feature map (channel)
plt.figure(figsize=(8, 8))
for i in range(64):  # Display the first 64 channels
    plt.subplot(8, 8, i+1)
    plt.imshow(feature_maps[0, :, :, i], cmap='gray')
    plt.axis('off')
plt.show()