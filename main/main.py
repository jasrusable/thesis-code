import os
import cv2

from detectors import SIFTDetector, ORBDetector
from matchers import BruteForceMatcher, FLANNMatcher
from thing import Thing


directory = '../images/model_1'

test_image = cv2.imread(os.path.join(directory, 'object.png'), 0)          # queryImage
train_image = cv2.imread(os.path.join(directory, 'scene.png'), 0) # trainImage

my_thing = Thing(
    test_image=test_image,
    train_image=train_image,
    detector=ORBDetector(),
    matcher=BruteForceMatcher(),
)

my_thing.detect_and_match()
my_thing.plot()
