from cv2 import imread


class Image(object):
    def __init__(self, file_path):
        self._file_path = file_path
        self._cv_image = imread(self._file_path, 0)
