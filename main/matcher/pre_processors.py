import cv2


class PreProcessor(object):
    def __init__(self):
        pass

    def process(self, cv_image):
        raise NotImplementedError(
            'This method is to be implemented in a subclass.'
        )


class Smoother(PreProcessor):
    def __init__(self):
        pass

    def process(self, cv_image):
        raise NotImplementedError(
            'This method is to be implemented in a subclass.'
        )


class AveragingSmoother(Smoother):
    def __init__(self, kernel_x, kernel_y):
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y

    def process(self, cv_image):
        return cv2.blur(
            cv_image, (self.kernel_x, self.kernel_y)
            )


class GaussianSmoother(Smoother):
    def __init__(self, kernel_x, kernel_y, sigma_x=0, sigma_y=0):
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def process(self, cv_image):
        return cv2.GaussianBlur(
            cv_image,
            (self.kernel_x, self.kernel_y),
            self.sigma_x,
            self.sigma_y
            )
