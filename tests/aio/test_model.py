import time

import numpy as np
import pytest

from tempo.aio.model import Model
from tempo.aio.utils import model
from tempo.errors import InvalidUserFunction
from tempo.serve.metadata import ModelFramework


async def test_invalid_model():
    with pytest.raises(InvalidUserFunction):

        @model(name="invalid-model", platform=ModelFramework.Custom)
        def _custom_model(payload: np.ndarray) -> np.ndarray:
            return payload


async def test_model(custom_model: Model):
    payload = np.array([1, 2, 3])
    res = await custom_model(payload)

    assert res == payload.sum(keepdims=True)


async def test_model_remote(sklearn_model):
    time.sleep(2)

    x_input = np.array([[1, 2, 3, 4]])
    y_pred = await sklearn_model(x_input)

    np.testing.assert_allclose(y_pred, [[0, 0, 0.99]], atol=1e-2)
