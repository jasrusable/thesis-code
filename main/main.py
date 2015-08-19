import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

import my_homogrophy
from detectors import SIFTDetector, ORBDetector
from matchers import BruteForceMatcher, FLANNMatcher

class Thing(object):
	def __init__(self, test_image, train_image, detector, matcher, detect_now=False, match_now=False):
		self.test_image = test_image
		self.train_image = train_image
		self.detector = detector
		self.matcher = matcher
		if detect_now:
			self.detect()
		if match_now:
			self.match()

	def detect(self):
		self.test_keypoints, self.test_descriptors = self.detector.compute(self.test_image)
		self.train_keypoints, self.train_descriptors = self.detector.compute(self.train_image)

	def match(self):
		self.matches = self.matcher.compute(self.test_descriptors, self.train_descriptors)

	def plot(self):
		img3 = my_homogrophy.get_homogrophy(
			self.test_image, 
			self.train_image, 
			self.test_keypoints, 
			self.train_keypoints, 
			self.matches
		)
		plt.imshow(img3, 'gray')
		plt.show()


directory = '../images/model_1'

object_image = cv2.imread(os.path.join(directory, 'object.png'), 0)          # queryImage
scene_image = cv2.imread(os.path.join(directory, 'scene.png'), 0) # trainImage

my_SIFT_detector = SIFTDetector()
my_ORB_detector = ORBDetector()
my_FLANN_matcher = FLANNMatcher()
my_brute_force_matcher = BruteForceMatcher()

my_thing = Thing(
	test_image=object_image,
	train_image=scene_image,
	detector=my_ORB_detector,
	matcher=my_brute_force_matcher,
)

my_thing.detect()
my_thing.match()
