import cv2
import os


class Image(object):
	def __init__(self, path):
		assert(os.path.exists(path))
		self.path = path
		self.cv_image = cv2.imread(path, 0)
		
class FeaturizedImage(Image):
	def __init__(self, path, keypoints, keypoint_descriptors):
		Image.__init__(self, path)
		self.keypoints = keypoints
		self.keypoint_descriptors = keypoint_descriptors
