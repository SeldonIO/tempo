import numpy as np

from tempo.aio.model import Model
from tempo.serve.loader import load, save


async def test_pipeline(inference_pipeline):
    x_input = np.array([[0.5, 2, 3, 4]])
    y_pred = await inference_pipeline(payload=x_input)

    np.testing.assert_allclose(y_pred, [[1.0]], atol=1e-2)


async def test_pipeline_remote(inference_pipeline):
    x_input = np.array([[0.5, 2, 3, 4]])
    y_pred = await inference_pipeline.predict(payload=x_input)

    np.testing.assert_allclose(y_pred, [[1.0]], atol=1e-2)


async def test_save(inference_pipeline):
    save(inference_pipeline, save_env=False)

    loaded_pipeline = load(inference_pipeline.details.local_folder)

    # Ensure models are exported as async
    assert len(inference_pipeline.models.__dict__) == len(loaded_pipeline.models.__dict__)
    for model in loaded_pipeline.models.values():
        assert isinstance(model, Model)
