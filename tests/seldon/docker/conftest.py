import time
from typing import Generator

import docker
import pytest

from tempo import Model, Pipeline
from tempo.seldon import SeldonDockerRuntime
from tempo.serve.loader import save
from tempo.serve.metadata import RuntimeOptions


@pytest.fixture
def runtime() -> SeldonDockerRuntime:
    return SeldonDockerRuntime(RuntimeOptions())


@pytest.fixture
def sklearn_model(sklearn_model: Model, runtime: SeldonDockerRuntime) -> Generator[Model, None, None]:
    runtime.deploy(sklearn_model)
    runtime.wait_ready(sklearn_model, timeout_secs=60)

    yield sklearn_model

    try:
        runtime.undeploy(sklearn_model)
    except docker.errors.NotFound:
        # TODO: Should undeploy be idempotent as well?
        # Ignore if the model has already been undeployed
        pass


@pytest.fixture
def xgboost_model(xgboost_model: Model, runtime: SeldonDockerRuntime) -> Generator[Model, None, None]:
    runtime.deploy(xgboost_model)
    runtime.wait_ready(xgboost_model, timeout_secs=60)

    yield xgboost_model

    try:
        runtime.undeploy(xgboost_model)
    except docker.errors.NotFound:
        # TODO: Should undeploy be idempotent as well?
        # Ignore if the model has already been undeployed
        pass


@pytest.fixture
def cifar10_model(cifar10_model: Model, runtime: SeldonDockerRuntime) -> Generator[Model, None, None]:
    runtime.deploy(cifar10_model)
    runtime.wait_ready(cifar10_model, timeout_secs=60)

    yield cifar10_model

    runtime.undeploy(cifar10_model)


@pytest.fixture
def inference_pipeline(
    inference_pipeline: Pipeline,
    runtime: SeldonDockerRuntime,
) -> Generator[Pipeline, None, None]:

    # NOTE: Need to re-save the pipeline so that it knows about the runtime
    save(inference_pipeline, save_env=True)
    runtime.deploy(inference_pipeline)
    runtime.wait_ready(inference_pipeline, timeout_secs=60)
    # TODO: Fix wait_ready for pipelines
    time.sleep(8)

    yield inference_pipeline

    try:
        runtime.undeploy(inference_pipeline)
    except docker.errors.NotFound:
        # TODO: Should undeploy be idempotent as well?
        # Ignore if the model has already been undeployed
        pass
