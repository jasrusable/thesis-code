import numpy as np
from matplotlib import pyplot as plt
from cv2 import findHomography, perspectiveTransform
from cv2 import polylines, drawMatches, LINE_AA
from cv2 import LINE_AA, RANSAC


class Thing(object):
    def __init__(self, test_image, train_image, detector, matcher, detect_now=False, match_now=False):
        self._test_image = test_image
        self._train_image = train_image
        if detector:
            self.set_detector(detector)
        if matcher:
            self.set_matcher(matcher)
        self._test_keypoints = None
        self._test_descriptors = None
        self._train_keypoints = None
        self._train_descriptors = None
        self._matches = None
        if detect_now:
            self.detect()
        if match_now:
            self.match()

    def set_detector(self, detector):
        self._detector = detector

    def set_matcher(self, matcher):
        self._matcher = matcher

    def detect(self):
        self._test_keypoints, self._test_descriptors = self._detector.detect(self._test_image._cv_image)
        self._train_keypoints, self._train_descriptors = self._detector.detect(self._train_image._cv_image)

    def match(self):
        self._matches = self._matcher.match(self._test_descriptors, self._train_descriptors)

    def detect_and_match(self):
        self.detect()
        self.match()

    def homogrophy(self):
        src_pts = np.float32([self._test_keypoints[match.queryIdx].pt for match in self._matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self._train_keypoints[match.trainIdx].pt for match in self._matches]).reshape(-1, 1, 2)
        M, mask = findHomography(src_pts, dst_pts, RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = self._test_image._cv_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = perspectiveTransform(pts, M)
        scene_image = polylines(self._train_image._cv_image, [np.int32(dst)], True, 255, 3, LINE_AA)
        draw_params = dict(
            matchColor = (0, 255, 0), # draw matches in green color
            singlePointColor = None,
            matchesMask = matchesMask, # draw only inliers
            flags = 2
        )
        img3 = drawMatches(
            self._test_image._cv_image,
            self._test_keypoints,
            self._train_image._cv_image,
            self._train_keypoints,
            self._matches,
            None, 
            **draw_params
        )
        return img3

    def plot(self):
        img3 = self.homogrophy()
        plt.imshow(img3, 'gray')
        plt.show()
