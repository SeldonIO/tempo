import pytest
import numpy as np

from typing import Generator

from tempo.seldon import SeldonDockerRuntime
from tempo.serve.model import Model as _Model
from tempo.serve.metadata import ModelFramework, RuntimeOptions
from tempo.aio.model import Model
from tempo.aio.utils import model


@pytest.fixture
def runtime() -> SeldonDockerRuntime:
    return SeldonDockerRuntime(RuntimeOptions())


@pytest.fixture
def sklearn_model(
    sklearn_model: _Model, runtime: SeldonDockerRuntime
) -> Generator[Model, None, None]:
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

    yield model

    runtime.undeploy(model)


@pytest.fixture
def custom_model() -> Model:
    @model(name="custom-model", platform=ModelFramework.Custom)
    async def _custom_model(payload: np.ndarray) -> np.ndarray:
        return payload.sum(keepdims=True)

    return _custom_model
