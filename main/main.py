import cv2
from os.path import join

from detectors import SIFTDetector, ORBDetector
from matchers import BruteForceMatcher, FLANNMatcher
from images import TestImage, TrainImage
from thing import Thing
from camera import Camera
from perspective_center import PerspectiveCenter


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

train_image = TrainImage(
    file_path=join(directory, 'scene.png'),
    camera=blender_cam,
    perspective_center=PerspectiveCenter(
        x=2.49426,
        y=1.35657,
        z=-0.24957,
    )
)

my_thing = Thing(
    test_image=test_image,
    train_image=train_image,
    detector=ORBDetector(),
    matcher=BruteForceMatcher(),
)

my_thing.detect_and_match()
my_thing.plot()
