impocv2.rt cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('object.png', cv2.IMREAD_COLOR)

plt.imshow(img)

plt.show()
