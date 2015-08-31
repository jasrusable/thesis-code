import cv2


class Smoother(object):
    def __init__(self):
        pass

    def smooth(self, image):
        raise NotImplementedError(
            'This method is to be implemented in a subclass.'
        )


class AveragingSmoother(Smoother):
    def __init__(self, kernel_x, kernel_y):
        self._kernel_x = kernel_x
        self._kernel_y = kernel_y

    def smooth(self, image):
        x = self._kernel_x
        y = self._kernel_y
        return cv2.blur(image.cv_image, (x, y))
