import numpy as np

from tempo.aio.model import Model


async def test_model(custom_model: Model):
    payload = np.array([1, 2, 3])
    res = await custom_model(payload)

    assert res == payload.sum(keepdims=True)
