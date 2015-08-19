import cv2
import numpy as np


def get_homogrophy(object_image, scene_image, object_image_keypoints, scene_image_keypoints, matches):
    src_pts = np.float32([object_image_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([scene_image_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

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

    img3 = cv2.drawMatches(object_image, object_image_keypoints, scene_image, scene_image_keypoints, matches, None, **draw_params)
    return img3
