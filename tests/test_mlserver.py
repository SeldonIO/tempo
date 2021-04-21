import pytest
from mlserver.settings import ModelParameters, ModelSettings
from mlserver.types import InferenceRequest, RequestInput
from mlserver.utils import to_ndarray

from tempo import Model
from tempo.mlserver import InferenceRuntime
from tempo.serve.loader import save


@pytest.fixture
def model_settings(custom_model: Model) -> ModelSettings:
    save(custom_model, save_env=False)
    model_uri = custom_model.details.local_folder

    return ModelSettings(
        name="sum-model",
        parameters=ModelParameters(uri=model_uri),
    )


@pytest.fixture
def inference_request() -> InferenceRequest:
    return InferenceRequest(inputs=[RequestInput(name="input-0", shape=[4], data=[1, 2, 3, 4], datatype="FP32")])


@pytest.fixture
async def mlserver_runtime(model_settings: ModelSettings) -> InferenceRuntime:
    _runtime = InferenceRuntime(model_settings)
    await _runtime.load()

    return _runtime


def test_load(mlserver_runtime: InferenceRuntime):
    assert mlserver_runtime.ready
    assert isinstance(mlserver_runtime._model, Model)


async def test_predict(
    mlserver_runtime: InferenceRuntime,
    inference_request: InferenceRequest,
    custom_model: Model,
):
    res = await mlserver_runtime.predict(inference_request)

    assert len(res.outputs) == 1

    pipeline_input = to_ndarray(inference_request.inputs[0])
    custom_model.get_tempo().set_remote(False)  # ensure direct call to class does not try to do remote
    expected_output = custom_model(pipeline_input)

    pipeline_output = res.outputs[0].data

    assert expected_output.tolist() == pipeline_output
