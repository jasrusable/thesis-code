import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('sift.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray, kp,1 )

plt.imshow(img, 'gray')
plt.show()
