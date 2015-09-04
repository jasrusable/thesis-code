import cv2


class PreProcessor(object):
    def __init__(self):
        pass

    def process(self, image):
        raise NotImplementedError(
            'This method is to be implemented in a subclass.'
        )


class Smoother(PreProcessor):
    def __init__(self):
        pass

    def process(self, image):
        raise NotImplementedError(
            'This method is to be implemented in a subclass.'
        )


class AveragingSmoother(Smoother):
    def __init__(self, kernel_x=1, kernel_y=1):
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y

    def process(self, image):
        image.cv_image = cv2.blur(
            image.cv_image, (self.kernel_x, self.kernel_y)
            )
        return image


class GaussianSmoother(Smoother):
    def __init__(self, kernel_x=1, kernel_y=1, sigma_x=0, sigma_y=0):
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def process(self, image):
        image.cv_image = cv2.GaussianBlur(
            image.cv_image,
            (self.kernel_x, self.kernel_y),
            self.sigma_x,
            self.sigma_y
            )
        return image
