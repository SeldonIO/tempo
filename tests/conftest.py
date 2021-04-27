import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from tempo import Model, ModelFramework, Pipeline, model, pipeline, predictmethod
from tempo.kfserving import KFServingV1Protocol, KFServingV2Protocol
from tempo.seldon import SeldonProtocol
from tempo.serve.constants import MLServerEnvDeps
from tempo.serve.metadata import KubernetesOptions, RuntimeOptions
from tempo.serve.pipeline import PipelineModels

TESTS_PATH = os.path.dirname(__file__)
TESTDATA_PATH = os.path.join(TESTS_PATH, "testdata")
PIPELINE_LOCAL_DIR = os.path.join(TESTS_PATH, "artifacts/pipeline")


def pytest_collection_modifyitems(items):
    """
    Add pytest.mark.asyncio marker to every test.
    """
    for item in items:
        item.add_marker("asyncio")


@pytest.fixture
def pipeline_conda_yaml() -> str:
    condaPath = PIPELINE_LOCAL_DIR + "/conda.yaml"
    if not os.path.isfile(condaPath):
        with open(PIPELINE_LOCAL_DIR + "/conda.yaml.tmpl") as f:
            env = yaml.safe_load(f)
            path = Path(TESTS_PATH)
            parent = path.parent.absolute()
            pip_deps = {"pip": []}
            env["dependencies"].append(pip_deps)
            for dep in MLServerEnvDeps:
                pip_deps["pip"].append(dep)
            pip_deps["pip"].append("mlops-tempo @ file://" + str(parent))
            with open(condaPath, "w") as f2:
                yaml.safe_dump(env, f2)
    return condaPath


@pytest.fixture
def sklearn_model() -> Model:
    model_path = os.path.join(TESTDATA_PATH, "sklearn", "iris")
    return Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris",
        local_folder=model_path,
        protocol=SeldonProtocol(),
        runtime_options=RuntimeOptions(k8s_options=KubernetesOptions(namespace="production", replicas=1)),
    )


@pytest.fixture
def xgboost_model() -> Model:
    model_path = os.path.join(TESTDATA_PATH, "xgboost", "iris")
    return Model(
        name="test-iris-xgboost",
        platform=ModelFramework.XGBoost,
        uri="gs://seldon-models/xgboost/iris",
        local_folder=model_path,
        protocol=SeldonProtocol(),
    )


@pytest.fixture
def custom_model() -> Model:
    @model(
        name="custom-model",
        protocol=KFServingV2Protocol(),
        platform=ModelFramework.Custom,
    )
    def _custom_model(payload: np.ndarray) -> np.ndarray:
        return _custom_model.context.model(payload)

    @_custom_model.loadmethod
    def _load():
        _custom_model.context.model = lambda a: a.sum(keepdims=True)

    return _custom_model


@pytest.fixture
def inference_pipeline(sklearn_model: Model, xgboost_model: Model, pipeline_conda_yaml: str) -> Pipeline:
    @pipeline(
        name="inference-pipeline",
        models=PipelineModels(sklearn=sklearn_model, xgboost=xgboost_model),
        uri="gs://seldon-models/tempo/test_pipeline",
        local_folder=PIPELINE_LOCAL_DIR,
    )
    def _pipeline(payload: np.ndarray) -> np.ndarray:
        res1 = _pipeline.models.sklearn(payload)
        if res1[0][0] > 0.7:
            return res1
        else:
            return _pipeline.models.xgboost(payload)

    return _pipeline


@pytest.fixture
def cifar10_model() -> Model:
    model_path = os.path.join(TESTDATA_PATH, "tfserving", "cifar10", "resnet32")
    return Model(
        name="resnet32",
        platform=ModelFramework.Tensorflow,
        uri="gs://seldon-models/tfserving/cifar10/resnet32",
        local_folder=model_path,
        protocol=KFServingV1Protocol(),
    )


@pytest.fixture
def inference_pipeline_class(sklearn_model: Model, xgboost_model: Model):
    @pipeline(
        name="mypipeline",
        models=PipelineModels(sklearn=sklearn_model, xgboost=xgboost_model),
        local_folder=PIPELINE_LOCAL_DIR,
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
