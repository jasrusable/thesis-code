import cv2
from os.path import join

from detectors import SIFTDetector, ORBDetector
from matchers import BruteForceMatcher, FLANNMatcher
from images import TestImage, QueryImage
from thing import Thing
from camera import Camera


blender_cam = Camera(
    focal_length=0.035,
    description='Blender camera.',
)
iphone4s = Camera(
    focal_length=0.035,
    description='Iphone 4S camera.',
)

directory = '../images/model_1'

test_image = TestImage(
    file_path=join(directory, 'object.png'),
    camera=iphone4s,
)
query_image = QueryImage(
    file_path=join(directory, 'scene.png'),
    camera=blender_cam,
)

my_thing = Thing(
    test_image=test_image,
    query_image=query_image,
    detector=ORBDetector(),
    matcher=BruteForceMatcher(),
)

my_thing.detect_and_match()
my_thing.plot()
