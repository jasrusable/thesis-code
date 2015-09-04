from matcher.test_case import TestCase
from matcher.images import TestImage, QueryImage
from matcher.detectors import ORBDetector
from matcher.matchers import BruteForceMatcher
from matcher.pre_processors import AveragingSmoother
from matcher.pre_processor_case import PreProcessorCase
from matcher.parameters import IntegerParameter


test_image = TestImage(
    file_path='../images/model_1/object.png',
)
query_image = QueryImage(
    file_path='../images/model_1/scene.png',
)

test_case = TestCase(
    test_image=test_image,
    query_image=query_image,
    test_preprocessor_cases=[
        PreProcessorCase(
            preprocessor=AveragingSmoother(),
            parameters=[IntegerParameter(
                name='kernel_y',
                from_=1,
                to=8,
                interval=None
            )],
        ),
    ],
    detector=ORBDetector(),
    matcher=BruteForceMatcher(),
)

test_case.run_tests()

test = test_case.tests[0]
test.do_all()
print(test.plot())
