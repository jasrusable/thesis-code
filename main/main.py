import cv2
import numpy as np
from os.path import join
from matplotlib import pyplot as plt

from matcher import Session, Thing
from matcher import Camera, TestImage, QueryImage
from matcher import ORBDetector, SIFTDetector
from matcher import BruteForceMatcher, FLANNMatcher
from matcher import AveragingSmoother


session = Session()

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

my_thing_2 = Thing(
    test_image=test_image,
    query_image=query_image,
    detector=SIFTDetector(),
    matcher=FLANNMatcher(),
)

#my_thing.detect_and_match()
