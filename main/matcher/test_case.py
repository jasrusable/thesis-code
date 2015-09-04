import copy
import numpy
from .test import Test


class TestCase(object):
    def __init__(self, test_image, query_image, detector,
                matcher,
                test_preprocessor_cases=[]):
        self.test_image = test_image
        self.query_image = query_image
        self.detector = detector
        self.matcher = matcher
        self.test_preprocessor_cases = test_preprocessor_cases
        self.tests = []

    def run_tests(self):
        for preprocessor_case in self.test_preprocessor_cases:
            preprocessor = preprocessor_case.preprocessor
            parameters = preprocessor_case.parameters
            for parameter in parameters:
                assert (hasattr(preprocessor, parameter.name))
                range_ = numpy.arange(
                    parameter.from_,
                    parameter.to,
                    parameter.step
                )
                for val in range_:
                    setattr(preprocessor, parameter.name, val)
                    self.tests.append(Test(
                        test_image=self.test_image,
                        query_image=self.query_image,
                        detector=self.detector,
                        matcher=self.matcher,
                        test_preprocessors=[copy.deepcopy(preprocessor)]
                    ))
