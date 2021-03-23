import os

import numpy as np
import pytest

from tempo import ModelFramework, Model, Pipeline, pipeline, predictmethod

TESTS_PATH = os.path.dirname(__file__)
TESTDATA_PATH = os.path.join(TESTS_PATH, "testdata")


def pytest_collection_modifyitems(items):
    """
    Add pytest.mark.asyncio marker to every test.
    """
    for item in items:
        item.add_marker("asyncio")


@pytest.fixture
def sklearn_model() -> Model:
    model_path = os.path.join(TESTDATA_PATH, "sklearn", "iris")
    return Model(
        name="test-iris-sklearn",
        runtime=None,
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris",
        local_folder=model_path,
    )


@pytest.fixture
def xgboost_model() -> Model:
    model_path = os.path.join(TESTDATA_PATH, "xgboost", "iris")
    return Model(
        name="test-iris-xgboost",
        runtime=None,
        platform=ModelFramework.XGBoost,
        uri="gs://seldon-models/xgboost/iris",
        local_folder=model_path,
    )


@pytest.fixture
def inference_pipeline(sklearn_model: Model, xgboost_model: Model) -> Pipeline:
    @pipeline(
        name="inference-pipeline",
        runtime=None,
        models=[sklearn_model, xgboost_model],
        uri="gs://seldon-models/tempo/test",
    )
    def _pipeline(payload: np.ndarray) -> np.ndarray:
        res1 = sklearn_model(payload)
        if res1[0][0] > 0.7:
            return res1
        else:
            return xgboost_model(payload)

    _pipeline.save(save_env=False)

    return _pipeline


@pytest.fixture
def cifar10_model() -> Model:
    model_path = os.path.join(TESTDATA_PATH, "tfserving", "cifar10", "resnet32")
    return Model(
        name="resnet32",
        platform=ModelFramework.Tensorflow,
        uri="gs://seldon-models/tfserving/cifar10/resnet32",
        local_folder=model_path,
    )


@pytest.fixture
def inference_pipeline_class(sklearn_model: Model, xgboost_model: Model):
    @pipeline(
        name="mypipeline",
        models=[sklearn_model, xgboost_model],
    )
    class MyClass(object):
        def __init__(self):
            self.counter = 0

        @predictmethod
        def p(self, payload: np.ndarray) -> np.ndarray:
            self.counter += 1
            return payload.sum(keepdims=True)

        def get_counter(self):
            return self.counter

    myc = MyClass()
    return myc
