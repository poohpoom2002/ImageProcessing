import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('rock-stones-boulder-sea.jpg')
template = cv2.imread('big-orange-sun-4.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

target_height, target_width = image.shape[:2]
template = cv2.resize(template, (target_width, target_height))

colors = ('r','g','b')

image_normalized_hist = []
template_normalized_hist = []
image_cumulative_hist = []
template_cumulative_hist = []

for i,color in enumerate(colors):
    image_temp = cv2.calcHist([image],[i],None,[256],[0,256])
    template_temp = cv2.calcHist([template],[i],None,[256],[0,256])

    normalized_image_hist = image_temp / image_temp.sum()
    normalized_template_hist = template_temp / template_temp.sum()

    cumulative_image_hist = np.cumsum(normalized_image_hist)
    cumulative_template_hist = np.cumsum(normalized_template_hist)

    image_normalized_hist.append(normalized_image_hist)
    template_normalized_hist.append(normalized_template_hist)
    image_cumulative_hist.append(cumulative_image_hist)
    template_cumulative_hist.append(cumulative_template_hist)

lookup_table = np.zeros((256), dtype=np.uint8)
for i,color in enumerate(colors):
    for src_bin in range(256):
        diff = np.abs(image_cumulative_hist[i][src_bin] - template_cumulative_hist[i])
        lookup_table[src_bin] = np.argmin(diff)

matched_image = cv2.LUT(image, lookup_table)

matched_image_normalized_hist = []
matched_image_cumulative_hist = []

for i,color in enumerate(colors):
    matched_image_hist = cv2.calcHist([matched_image],[i],None,[256],[0,256])

    normalized_matched_image_hist = matched_image_hist / matched_image_hist.sum()
    matched_image_normalized_hist.append(normalized_matched_image_hist)

    cumulative_matched_image_hist = np.cumsum(normalized_matched_image_hist)
    matched_image_cumulative_hist.append(cumulative_matched_image_hist)

plt.subplot(3, 3, 1)
plt.imshow(image)
plt.title('Source Image')

plt.subplot(3, 3, 4)
plt.imshow(template)
plt.title('Target Image')

plt.subplot(3, 3, 7)
plt.imshow(matched_image)
plt.title('Matched Image')

plt.subplot(3, 3, 2)
plt.plot(image_normalized_hist[0], color='red')
plt.plot(image_normalized_hist[1], color='green')
plt.plot(image_normalized_hist[2], color='blue')
plt.title('image normalize hist')

plt.subplot(3, 3, 5)
plt.plot(template_normalized_hist[0], color='red')
plt.plot(template_normalized_hist[1], color='green')
plt.plot(template_normalized_hist[2], color='blue')
plt.title('template normalize hist')

plt.subplot(3, 3, 8)
plt.plot(matched_image_normalized_hist[0], color='red')
plt.plot(matched_image_normalized_hist[1], color='green')
plt.plot(matched_image_normalized_hist[2], color='blue')
plt.title('matched image normalize hist')

plt.subplot(3, 3, 3)
plt.plot(image_cumulative_hist[0], color='red')
plt.plot(image_cumulative_hist[1], color='green')
plt.plot(image_cumulative_hist[2], color='blue')
plt.title('image normalize hist')

plt.subplot(3, 3, 6)
plt.plot(template_cumulative_hist[0], color='red')
plt.plot(template_cumulative_hist[1], color='green')
plt.plot(template_cumulative_hist[2], color='blue')
plt.title('template normalize hist')

plt.subplot(3, 3, 9)
plt.plot(matched_image_cumulative_hist[0], color='red')
plt.plot(matched_image_cumulative_hist[1], color='green')
plt.plot(matched_image_cumulative_hist[2], color='blue')
plt.title('matched image normalize hist')

plt.tight_layout()
plt.show()