import cv2

from images import Image, FeaturizedImage


class Sift(object):
    def get_sifted_image(self, image):
        cv_sift = cv2.xfeatures2d.SIFT_create(450)
        keypoints, keypoint_descriptors = (
            cv_sift.detectAndCompute(image.cv_image, None)
        )
        featurized_image = FeaturizedImage(
            path=image.path,
            keypoints=keypoints,
            keypoint_descriptors=keypoint_descriptors,
        )
        return featurized_image


class JasonCV(object):
    def __init__(self):
        self.sift = Sift()

    def run_sift_homogrophy(image_1, image_2):
        sifted_image_1 = self.get_sifted_image(image_1)
        sifted_image_2 = self.get_sifted_image(image_2)
        matches = self.run_flan(sifted_image_1, sifted_image_2)

        good_matches = []
        for m, n in matches:
            if m.distance / n.distance < 0.7 * :
                good_matches.append(m)

        assert len(good_matches) > 10

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    

    def run_flan(self, image_1, image_2):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(
            algorithm = FLANN_INDEX_KDTREE, 
            trees = 5,
        )
        search_params = dict(
            checks = 50
        )
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        des1 = image_1.keypoint_descriptors
        des2 = image_2.keypoint_descriptors
        matches = flann.knnMatch(des1, des2, k=2)
        return matches
