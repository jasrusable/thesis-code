import os
import cv2

from detectors import SIFTDetector, ORBDetector
from matchers import BruteForceMatcher, FLANNMatcher
from image import Image
from thing import Thing


directory = '../images/model_1'

test_image = Image(file_path=os.path.join(directory, 'object.png'))          # queryImage
train_image = Image(file_path=os.path.join(directory, 'scene.png')) # trainImage

my_thing = Thing(
    test_image=test_image,
    train_image=train_image,
    detector=ORBDetector(),
    matcher=BruteForceMatcher(),
)

my_thing.detect_and_match()
