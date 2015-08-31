import numpy as np
from matplotlib import pyplot as plt
from cv2 import findHomography, perspectiveTransform
from cv2 import polylines, drawMatches
from cv2 import LINE_AA, RANSAC


class TestCase(object):
    def __init__(self, test_image, query_image, detector,
                matcher, detect_now=False, match_now=False,
                test_preprocessors=[], query_preprocessors=[]):
        self.test_image = test_image
        self.query_image = query_image
        self.detector = detector
        self.matcher = matcher
        self.test_keypoints = None
        self.test_descriptors = None
        self.query_keypoints = None
        self.query_descriptors = None
        self.matches = None
        self.test_preprocessors = test_preprocessors
        self.query_preprocessors = query_preprocessors
        if detect_now:
            self.detect()
        if match_now:
            self.match()

    def preprocess(self):
        for preprocesser in self.test_preprocessors:
            self.test_image.cv_image = preprocesser.process(self.test_image.cv_image)

        for preprocesser in self.query_preprocessors:
            self.query_image.cv_image = preprocesser.process(self.query_image.cv_image)

    def detect(self):
        self.test_keypoints, self.test_descriptors = (
            self.detector.detect(self.test_image.cv_image)
        )
        self.query_keypoints, self.query_descriptors = (
            self.detector.detect(self.query_image.cv_image)
        )

    def match(self):
        self.matches = self.matcher.match(
            self.test_descriptors, self.query_descriptors
        )

    def do_all(self):
        self.preprocess()
        self.detect()
        self.match()

    def homogrophy(self):
        src_pts = np.float32([self.test_keypoints[match.queryIdx].pt
                            for match in self.matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.query_keypoints[match.trainIdx].pt
                            for match in self.matches]).reshape(-1, 1, 2)
        M, mask = findHomography(src_pts, dst_pts, RANSAC, 10.0)
        matchesMask = mask.ravel().tolist()
        h, w = self.test_image.cv_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = perspectiveTransform(pts, M)
        scene_image = polylines(
            self.query_image.cv_image, [np.int32(dst)], True, 255, 3, LINE_AA
        )
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor = None,
            matchesMask = matchesMask,
            flags=2,
        )
        return draw_params

    def plot(self):
        draw_params = self.homogrophy()
        img3 = drawMatches(
            self.test_image.cv_image,
            self.test_keypoints,
            self.query_image.cv_image,
            self.query_keypoints,
            self.matches,
            None,
            **draw_params
        )
        plt.imshow(img3, 'gray')
        plt.show()
