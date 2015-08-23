from os.path import isfile
from cv2 import imread


class Image(object):
    def __init__(self, file_path, camera=None):
        assert isfile(file_path)
        self.file_path = file_path
        self.cv_image = imread(self.file_path, 0)
        self.camera = camera

class TestImage(Image):
    def __init__(self, file_path, camera=None):
        Image.__init__(self, file_path, camera)

class QueryImage(Image):
    def __init__(self, file_path, camera):
        Image.__init__(self, file_path, camera)
