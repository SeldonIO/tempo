import uuid
import pytest
import os
import time
import docker
import numpy as np

from typing import Generator

from kubernetes import client, config

from tempo.serve.metadata import ModelFramework, KubernetesOptions
from tempo.serve.model import Model
from tempo.serve.pipeline import Pipeline
from tempo.serve.utils import pipeline
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.serve.utils import predictmethod

TESTS_PATH = os.path.dirname(os.path.dirname(__file__))
EXAMPLES_PATH = os.path.join(TESTS_PATH, "examples")

K8S_NAMESPACE_PREFIX = "test-tempo-"


def pytest_collection_modifyitems(items):
    """
    Add pytest.mark.asyncio marker to every test.
    """
    for item in items:
        item.add_marker("asyncio")


@pytest.fixture
def docker_runtime() -> Generator[SeldonDockerRuntime, None, None]:
    runtime = SeldonDockerRuntime()

    yield runtime


@pytest.fixture
def docker_runtime_v2() -> Generator[SeldonDockerRuntime, None, None]:
    runtime = SeldonDockerRuntime(protocol=KFServingV2Protocol())

    yield runtime


@pytest.fixture
def k8s_namespace() -> Generator[str, None, None]:
    unique_id = str(uuid.uuid4().fields[-1])[:5]
    ns_name = f"{K8S_NAMESPACE_PREFIX}{unique_id}"

    config.load_kube_config()
    core_api = client.CoreV1Api()

    namespace = client.V1Namespace(metadata=client.V1ObjectMeta(name=ns_name))
    created_namespace = core_api.create_namespace(namespace)

    yield ns_name

    core_api.delete_namespace(ns_name)


@pytest.fixture
def k8s_runtime(k8s_namespace: str) -> SeldonKubernetesRuntime:
    return SeldonKubernetesRuntime(
        k8s_options=KubernetesOptions(namespace=k8s_namespace)
    )


@pytest.fixture
def k8s_sklearn_model(
    sklearn_model: Model, k8s_runtime: SeldonKubernetesRuntime
) -> Generator[Model, None, None]:
    sklearn_model._runtime = k8s_runtime

    sklearn_model.deploy()
    sklearn_model.wait_ready(timeout_secs=60)

    yield sklearn_model

    sklearn_model.undeploy()


@pytest.fixture
def sklearn_iris_path() -> str:
    return os.path.join(EXAMPLES_PATH, "sklearn", "iris")


@pytest.fixture
def xgboost_iris_path() -> str:
    return os.path.join(EXAMPLES_PATH, "xgboost", "iris")


@pytest.fixture
def sklearn_model(sklearn_iris_path: str, docker_runtime: SeldonDockerRuntime) -> Model:
    return Model(
        name="test-iris-sklearn",
        runtime=docker_runtime,
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris",
        local_folder=sklearn_iris_path,
    )


@pytest.fixture
def xgboost_model(xgboost_iris_path: str, docker_runtime: SeldonDockerRuntime) -> Model:
    return Model(
        name="test-iris-xgboost",
        runtime=docker_runtime,
        platform=ModelFramework.XGBoost,
        uri="gs://seldon-models/sklearn/iris",
        local_folder=xgboost_iris_path,
    )


@pytest.fixture
def inference_pipeline(
    sklearn_model: Model, xgboost_model: Model, docker_runtime_v2: SeldonDockerRuntime
) -> Generator[Pipeline, None, None]:
    @pipeline(
        name="inference-pipeline",
        runtime=docker_runtime_v2,
        models=[sklearn_model, xgboost_model],
    )
    def _pipeline(payload: np.ndarray) -> np.ndarray:
        res1 = sklearn_model(payload)
        if res1[0][0] > 0.7:
            return res1
        else:
            return xgboost_model(payload)

    _pipeline.save()
    _pipeline.deploy()
    time.sleep(2)

    yield _pipeline

    try:
        _pipeline.undeploy()
    except docker.errors.NotFound:
        # Ignore if the pipeline was already undeployed
        pass


@pytest.fixture
def inference_pipeline_v2(
    sklearn_model: Model, xgboost_model: Model, docker_runtime_v2: SeldonDockerRuntime
) -> Generator[Pipeline, None, None]:
    @pipeline(
        name="inference-pipeline",
        runtime=docker_runtime_v2,
        models=[sklearn_model, xgboost_model],
    )
    def _pipeline(payload: np.ndarray) -> np.ndarray:
        res1 = sklearn_model(payload)
        if res1[0][0] > 0.7:
            return res1
        else:
            return xgboost_model(payload)

    _pipeline.deploy()
    time.sleep(2)

    yield _pipeline

    try:
        _pipeline.undeploy()
    except docker.errors.NotFound:
        # Ignore if the pipeline was already undeployed
        pass


@pytest.fixture
def inference_pipeline_v3(
    sklearn_model: Model, xgboost_model: Model, docker_runtime_v2: SeldonDockerRuntime
):
    @pipeline(
        name="mypipeline",
        runtime=docker_runtime_v2,
        models=[sklearn_model, xgboost_model],
    )
    class MyClass(object):
        def __init__(self):
            self.counter = 0

        @predictmethod
        def p(self, payload: np.ndarray) -> np.ndarray:
            self.counter += 1
            res1 = sklearn_model(payload)
            if res1[0][0] > 0.7:
                return res1
            else:
                return xgboost_model(payload)

        def get_counter(self):
            return self.counter

    myc = MyClass()
    myc.pipeline.deploy()

    time.sleep(2)

    yield myc

    try:
        myc.undeploy()
    except docker.errors.NotFound:
        # Ignore if the pipeline was already undeployed
        pass
