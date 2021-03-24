import time
from typing import Generator

import docker
import pytest

from tempo import Model, Pipeline
from tempo.kfserving import KFServingV1Protocol, KFServingV2Protocol
from tempo.seldon import SeldonDockerRuntime


@pytest.fixture
def runtime() -> SeldonDockerRuntime:
    return SeldonDockerRuntime()


@pytest.fixture
def runtime_v2() -> SeldonDockerRuntime:
    return SeldonDockerRuntime(protocol=KFServingV2Protocol())


@pytest.fixture
def runtime_kfserving() -> SeldonDockerRuntime:
    return SeldonDockerRuntime(protocol=KFServingV1Protocol())


@pytest.fixture
def sklearn_model(sklearn_model: Model, runtime: SeldonDockerRuntime) -> Generator[Model, None, None]:
    sklearn_model.set_runtime(runtime)
    sklearn_model.deploy()
    sklearn_model.wait_ready(timeout_secs=60)

    yield sklearn_model

    try:
        sklearn_model.undeploy()
    except docker.errors.NotFound:
        # TODO: Should undeploy be idempotent as well?
        # Ignore if the model has already been undeployed
        pass


@pytest.fixture
def xgboost_model(xgboost_model: Model, runtime: SeldonDockerRuntime) -> Generator[Model, None, None]:
    xgboost_model.set_runtime(runtime)
    xgboost_model.deploy()
    xgboost_model.wait_ready(timeout_secs=60)

    yield xgboost_model

    try:
        xgboost_model.undeploy()
    except docker.errors.NotFound:
        # TODO: Should undeploy be idempotent as well?
        # Ignore if the model has already been undeployed
        pass


@pytest.fixture
def cifar10_model(cifar10_model: Model, runtime_kfserving: SeldonDockerRuntime) -> Generator[Model, None, None]:
    cifar10_model.set_runtime(runtime_kfserving)
    cifar10_model.deploy()
    cifar10_model.wait_ready(timeout_secs=60)

    yield cifar10_model

    cifar10_model.undeploy()


@pytest.fixture
def inference_pipeline(
    inference_pipeline: Pipeline,
    runtime: SeldonDockerRuntime,
    runtime_v2: SeldonDockerRuntime,
) -> Generator[Pipeline, None, None]:
    inference_pipeline.set_runtime(runtime_v2)

    for model in inference_pipeline._models:
        model.set_runtime(runtime)

    # NOTE: Need to re-save the pipeline so that it knows about the runtime
    inference_pipeline.save(save_env=False)
    inference_pipeline.deploy()
    inference_pipeline.wait_ready(timeout_secs=60)
    # TODO: Fix wait_ready for pipelines
    time.sleep(3)

    yield inference_pipeline

    try:
        inference_pipeline.undeploy()
    except docker.errors.NotFound:
        # TODO: Should undeploy be idempotent as well?
        # Ignore if the model has already been undeployed
        pass
