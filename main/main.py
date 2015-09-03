import cv2
import numpy as np
from os.path import join
from matplotlib import pyplot as plt

from matcher import Session, TestCase
from matcher import Camera, TestImage, QueryImage
from matcher import ORBDetector, SIFTDetector
from matcher import BruteForceMatcher, FLANNMatcher
from matcher import AveragingSmoother, GaussianSmoother


session = Session()

blender_cam = Camera(
    focal_length=0.035,
    description='Blender camera.',
)
iphone4s = Camera(
    focal_length=0.035,
    description='Iphone 4S camera.',
)

test_image = TestImage(
    file_path='../images/model_1/object.png',
    camera=iphone4s,
)
query_image = QueryImage(
    file_path='../images/model_1/scene.png',
    camera=blender_cam,
)

my_thing = TestCase(
    test_image=test_image,
    query_image=query_image,
    test_preprocessors=[
        AveragingSmoother(kernel_x=1, kernel_y=1),
    ],
    query_preprocessors=[
    ],
    detector=ORBDetector(),
    matcher=BruteForceMatcher(),
)

my_thing.do_all()
my_thing.plot()
