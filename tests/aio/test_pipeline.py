import time
import numpy as np

from tempo.aio.pipeline import Pipeline


async def test_pipeline(inference_pipeline: Pipeline):
    time.sleep(2)

    x_input = np.array([[0.5, 2, 3, 4]])
    y_pred = await inference_pipeline(payload=x_input)

    np.testing.assert_allclose(y_pred, [[1.0]], atol=1e-2)
