import os
import time
from typing import Generator

import docker
import numpy as np
import pytest

from tempo.aio.model import Model
from tempo.aio.pipeline import Pipeline
from tempo.aio.utils import model, pipeline
from tempo.seldon import SeldonDockerRuntime
from tempo.serve.metadata import DockerOptions, ModelFramework
from tempo.serve.model import Model as _Model
from tempo.serve.pipeline import PipelineModels


@pytest.fixture
def runtime() -> SeldonDockerRuntime:
    return SeldonDockerRuntime(DockerOptions())


@pytest.fixture
def sklearn_model(sklearn_model: _Model, runtime: SeldonDockerRuntime) -> Generator[Model, None, None]:
    model = Model(
        name=sklearn_model.details.name,
        platform=sklearn_model.details.platform,
        uri=sklearn_model.details.uri,
        local_folder=sklearn_model.details.local_folder,
        protocol=sklearn_model.model_spec.protocol,
        runtime_options=sklearn_model.model_spec.runtime_options,
    )

    runtime.deploy(model)
    runtime.wait_ready(model)
    time.sleep(2)

    yield model

    try:
        runtime.undeploy(model)
    except docker.errors.NotFound:
        # TODO: Should undeploy be idempotent as well?
        # Ignore if the model has already been undeployed
        pass


@pytest.fixture
def custom_model() -> Model:
    @model(name="custom-model", platform=ModelFramework.Custom)
    async def _custom_model(payload: np.ndarray) -> np.ndarray:
        return payload.sum(keepdims=True)

    return _custom_model


@pytest.fixture
def inference_pipeline(
    sklearn_model, runtime: SeldonDockerRuntime, pipeline_conda_yaml: str
) -> Generator[Pipeline, None, None]:
    @pipeline(
        name="inference-pipeline",
        models=PipelineModels(sklearn=sklearn_model),
        local_folder=os.path.dirname(pipeline_conda_yaml),
    )
    async def _pipeline(payload: np.ndarray) -> np.ndarray:
        res1 = await _pipeline.models.sklearn(payload)
        if res1[0][0] > 0.7:
            return res1

        return res1.sum(keepdims=True)

    _pipeline.save(save_env=True)
    runtime.deploy(_pipeline)
    runtime.wait_ready(_pipeline)
    time.sleep(8)

    yield _pipeline

    try:
        runtime.undeploy(_pipeline)
    except docker.errors.NotFound:
        # TODO: Should undeploy be idempotent as well?
        # Ignore if the model has already been undeployed
        pass
