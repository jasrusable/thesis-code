import itertools
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
        preprocessors_by_parameters = []
        for preprocessor_case in self.test_preprocessor_cases:
            preprocessor = preprocessor_case.preprocessor
            parameters = preprocessor_case.parameters
            for parameter in parameters:
                temp = []
                for i in parameter.range:
                    new_preprocessor = copy.deepcopy(preprocessor)
                    setattr(new_preprocessor, parameter.name, i)
                    temp.append(new_preprocessor)
                preprocessors_by_parameters.append(temp)
            for c in itertools.product(*preprocessors_by_parameters):
                self.tests.append(
                    Test(
                        test_image=copy.deepcopy(self.test_image),
                        query_image=copy.deepcopy(self.query_image),
                        detector=self.detector,
                        matcher=self.matcher,
                        test_preprocessors=c,
                    )
                )
