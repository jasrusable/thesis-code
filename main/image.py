from os.path import isfile
from cv2 import imread

from perspective_center import PerspectiveCenter


class Image(object):
    def __init__(self, file_path, camera=None, perspective_center=None):
        assert isfile(file_path)
        self._file_path = file_path
        self._cv_image = imread(self._file_path, 0)
        self._camera = camera
        self._perspective_center = perspective_center
