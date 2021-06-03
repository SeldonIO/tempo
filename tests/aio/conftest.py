import pytest
import numpy as np

from tempo.serve.model import Model as _Model
from tempo.serve.metadata import ModelFramework
from tempo.aio.model import Model
from tempo.aio.utils import model


@pytest.fixture
def sklearn_model(sklearn_model: _Model) -> Model:
    return Model(
        name=sklearn_model.name,
        platform=sklearn_model.details.platform,
        uri=sklearn_model.details.uri,
        local_folder=sklearn_model.details.local_folder,
        protocol=sklearn_model.model_spec.protocol,
        runtime_options=sklearn_model.model_spec.runtime_options,
    )


@pytest.fixture
def custom_model() -> Model:
    @model(name="custom-model", platform=ModelFramework.Custom)
    async def _custom_model(payload: np.ndarray) -> np.ndarray:
        return payload.sum(keepdims=True)

    return _custom_model
