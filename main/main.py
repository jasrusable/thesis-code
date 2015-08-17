import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0

directory = '../images/model_1'

object_image = cv2.imread(os.path.join(directory, 'object.png'), 0)          # queryImage
scene_image = cv2.imread(os.path.join(directory, 'scene.png'), 0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create(450)

# find the keypoints and descriptors with SIFT
object_image_keypoints, object_image_descriptors = sift.detectAndCompute(object_image, None)
scene_image_keypoints, scene_image_descriptors = sift.detectAndCompute(scene_image, None)

index_params = dict(
    algorithm = FLANN_INDEX_KDTREE, 
    trees = 5,
)

search_params = dict(
    checks = 50
)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(object_image_descriptors, scene_image_descriptors, k=2)

# store all the good_matches matches as per Lowe's ratio test.
good_matches = []
for object_image_match, scene_image_match in matches:
    if object_image_match.distance < 0.7 * scene_image_match.distance:
        good_matches.append(object_image_match)

assert len(good_matches) > MIN_MATCH_COUNT

src_pts = np.float32([object_image_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([scene_image_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

h, w = object_image.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)

scene_image = cv2.polylines(scene_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

draw_params = dict(
    matchColor = (0, 255, 0), # draw matches in green color
    singlePointColor = None,
    matchesMask = matchesMask, # draw only inliers
    flags = 2
)

img3 = cv2.drawMatches(object_image, object_image_keypoints, scene_image, scene_image_keypoints, good_matches, None, **draw_params)

plt.imshow(img3, 'gray')
plt.show()
