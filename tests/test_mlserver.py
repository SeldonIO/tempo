import copy
from inspect import iscoroutine

import pytest
from mlserver.settings import ModelSettings
from mlserver.types import InferenceRequest, RequestInput
from mlserver.utils import to_ndarray
from pytest_cases import fixture, parametrize_with_cases
from pytest_cases.common_pytest_lazy_values import is_lazy

from tempo.mlserver import InferenceRuntime
from tempo.serve.base import BaseModel

from .test_mlserver_cases import case_wrapped_class


@pytest.fixture
def inference_request() -> InferenceRequest:
    return InferenceRequest(inputs=[RequestInput(name="payload", shape=[4], data=[1, 2, 3, 4], datatype="FP32")])


@fixture
@parametrize_with_cases("model_settings")
async def mlserver_runtime(model_settings: ModelSettings) -> InferenceRuntime:
    if is_lazy(model_settings):
        # NOTE: Some times pytest-cases may return a "LazyValue"
        model_settings = model_settings.get(request_or_item=mlserver_runtime)

    _runtime = InferenceRuntime(model_settings)
    await _runtime.load()

    return _runtime


async def test_load(mlserver_runtime: InferenceRuntime):
    # NOTE: pytest-cases doesn't wait for async fixtures
    # TODO: Raise issue in pytest-cases repo
    mlserver_runtime = await mlserver_runtime
    assert mlserver_runtime.ready
    assert isinstance(mlserver_runtime._model, BaseModel)


async def test_predict(mlserver_runtime: InferenceRuntime, inference_request: InferenceRequest):
    # NOTE: pytest-cases doesn't wait for async fixtures
    # TODO: Raise issue in pytest-cases repo
    mlserver_runtime = await mlserver_runtime
    res = await mlserver_runtime.predict(inference_request)

    assert len(res.outputs) == 1

    pipeline_input = to_ndarray(inference_request.inputs[0])
    custom_model = copy.copy(mlserver_runtime._model)
    # Ensure direct call to class does not try to do remote
    custom_model.set_remote(False)
    expected_output = custom_model(payload=pipeline_input)
    if iscoroutine(expected_output):
        expected_output = await expected_output

    pipeline_output = res.outputs[0].data

    assert expected_output.tolist() == pipeline_output


async def test_load_wrapped_class(inference_pipeline_class, inference_request: InferenceRequest):
    pipeline_input = to_ndarray(inference_request.inputs[0])

    inference_pipeline_class(pipeline_input)
    assert inference_pipeline_class.counter == 1

    model_settings = case_wrapped_class(inference_pipeline_class)
    runtime = InferenceRuntime(model_settings)
    await runtime.load()

    assert inference_pipeline_class.counter == 1
    assert runtime._model._user_func.__self__.counter == 0  # type: ignore
