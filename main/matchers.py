from cv2 import BFMatcher, NORM_HAMMING
from cv2 import FlannBasedMatcher


class Matcher(object):
    def __init__(self):
        pass

    def compute(self, test_descriptors, train_descriptors):
        raise NotImplementedError('This method is to be implemented in a subclass.')

class BruteForceMatcher(Matcher):
    def __init__(self):
        Matcher.__init__(self)
        self.brute_force_matcher = BFMatcher(NORM_HAMMING, crossCheck=True)

    def __repr__(self):
        return ("BruteForceMatcher(brute_force_matcher={brute_force_matcher})"
            .format(brute_force_matcher=self.brute_force_matcher))

    def compute(self, test_descriptors, train_descriptors):
        matches = self.brute_force_matcher.match(test_descriptors, train_descriptors)
        matches = sorted(matches, key = lambda x:x.distance)
        return matches

class FLANNMatcher(Matcher):
    def __init__(self):
        Matcher.__init__(self)
        FLANN_INDEX_KDTREE = 0
        index_params = {
            'algorithm': FLANN_INDEX_KDTREE, 
            'trees': 5,
        }
        search_params = {
            'checks': 50,
        }
        self.FLANN = FlannBasedMatcher(index_params, search_params)

    def __repr__(self):
        return "FLANNMatcher(FLANN={FLANN})".format(FLANN=self.FLANN)

    def compute(self, test_descriptors, train_descriptors):
        matches = self.FLANN.knnMatch(test_descriptors, train_descriptors, k=2)
        #TODO: Where should this good_matches filter happen?
        good_matches = []
        # store all the good_matches matches as per Lowe's ratio test.
        for object_image_match, scene_image_match in matches:
            if object_image_match.distance < 0.7 * scene_image_match.distance:
                good_matches.append(object_image_match)
        return good_matches
