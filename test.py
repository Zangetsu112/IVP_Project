import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from forgery import *

image = cv2.imread('./all-mias/mdb001.pgm')
image2 = cv2.imread('./all-mias/mdb005.pgm')

grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
grey_image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
# plt.imshow(grey_image2)
# plt.show()

splice = extract_splice(grey_image)
splice2 = extract_splice(grey_image2)
all_splice = [splice, splice2]

image_new = add_splice(grey_image, all_splice)
image2_new = copy_paste(grey_image2, splice2)

def plot(subplots, r, c):
    plt.figure(figsize=(8, 8))
    for i in range(1, len(subplots)+1):
        plt.subplot(r, c, i)
        plt.imshow(subplots[i-1], cmap='gray')
    plt.show()

plot([grey_image, splice, image_new, grey_image-image_new], 2, 2)
# plot([grey_image2, splice2, image2_new, grey_image2-image2_new], 2, 2)
