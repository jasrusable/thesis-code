from cv2 import ORB_create
from cv2.xfeatures2d import SIFT_create


class Detector(object):
    def __init__(self):
        pass

    def compute(self, image):
        raise NotImplementedError('This method is to be implemented in a subclass.')

class ORBDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.ORB = ORB_create()

    def compute(self, image):
        keypoints = self.ORB.detect(image, None)
        keypoints, descriptors = self.ORB.compute(image, keypoints)
        return keypoints, descriptors

class SIFTDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.SIFT = SIFT_create()

    def compute(self, image):
        keypoints, descriptors = self.SIFT.detectAndCompute(image, None)
        return keypoints, descriptors
