from typing import List, Tuple

import numpy as np
import pytest

from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.serve.utils import model, predictmethod


#
# Test single input output model with type annotations
#
@pytest.mark.parametrize(
    "v2_input, expected",
    [
        (
            {
                "inputs": [
                    {
                        "name": "a",
                        "datatype": "FP64",
                        "shape": [1, 4],
                        "data": [1, 2, 3, 4],
                    }
                ]
            },
            {
                "model_name": "custom",
                "outputs": [
                    {
                        "name": "output0",
                        "datatype": "FP64",
                        "data": [1.0, 2.0, 3.0, 4.0],
                        "shape": [1, 4],
                    }
                ],
            },
        )
    ],
)
def test_custom_model(v2_input, expected):
    @model(
        name="custom",
        protocol=KFServingV2Protocol(),
        platform=ModelFramework.Custom,
    )
    def custom_model(a: np.ndarray) -> np.ndarray:
        return a

    response = custom_model.request(v2_input)
    assert response == expected


#
# Test lambda function
#
@pytest.mark.parametrize("input, expected", [(np.array([[0, 0, 0, 1]]), np.array([[0, 0, 1]]))])
def test_lambda(input, expected):
    model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.Custom,
        model_func=lambda x: np.array([[0, 0, 1]]),
    )

    response = model(input)
    np.testing.assert_allclose(response, expected, atol=1e-2)


#
# Test single input output model with type annotations
#


@pytest.mark.parametrize(
    "v2_input, expected",
    [
        (
            {
                "inputs": [
                    {
                        "name": "a",
                        "datatype": "FP64",
                        "shape": [1, 4],
                        "data": [1, 2, 3, 4],
                    }
                ]
            },
            {
                "model_name": "custom",
                "outputs": [
                    {
                        "name": "output0",
                        "datatype": "FP64",
                        "data": [1.0, 2.0, 3.0, 4.0],
                        "shape": [1, 4],
                    }
                ],
            },
        )
    ],
)
def test_custom_model_decorator_types(v2_input, expected):
    @model(
        name="custom",
        platform=ModelFramework.Custom,
        inputs=np.ndarray,
        outputs=np.ndarray,
    )
    def custom_model_decorator_types(a):
        return a

    response = custom_model_decorator_types.request(v2_input)
    assert response == expected


#
# Test multi-headed input output model with type annotations with tuple return
#
@pytest.mark.parametrize(
    "v2_input, expected",
    [
        (
            {
                "inputs": [
                    {
                        "name": "a",
                        "datatype": "FP64",
                        "shape": [1, 4],
                        "data": [1, 2, 3, 4],
                    },
                    {"name": "b", "datatype": "FP64", "shape": [1, 2], "data": [4, 5]},
                ]
            },
            {
                "model_name": "multi-headed",
                "outputs": [
                    {
                        "name": "output0",
                        "datatype": "FP64",
                        "data": [1.0, 2.0, 3.0, 4.0],
                        "shape": [1, 4],
                    },
                    {
                        "name": "output1",
                        "datatype": "FP64",
                        "data": [4.0, 5.0],
                        "shape": [1, 2],
                    },
                ],
            },
        )
    ],
)
def test_custom_multiheaded_model_tuple(v2_input, expected):
    @model(
        name="multi-headed",
        platform=ModelFramework.Custom,
    )
    def custom_multiheaded_model_tuple(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return a, b

    response = custom_multiheaded_model_tuple.request(v2_input)
    assert response == expected


#
# Test multi-headed input output model with type annotations with list return
#
@pytest.mark.parametrize(
    "v2_input, expected",
    [
        (
            {
                "inputs": [
                    {
                        "name": "a",
                        "datatype": "FP64",
                        "shape": [1, 4],
                        "data": [1, 2, 3, 4],
                    },
                    {"name": "b", "datatype": "FP64", "shape": [1, 2], "data": [4, 5]},
                ]
            },
            {
                "model_name": "multi-headed",
                "outputs": [
                    {
                        "name": "output0",
                        "datatype": "FP64",
                        "data": [1.0, 2.0, 3.0, 4.0],
                        "shape": [1, 4],
                    },
                    {
                        "name": "output1",
                        "datatype": "FP64",
                        "data": [4.0, 5.0],
                        "shape": [1, 2],
                    },
                ],
            },
        )
    ],
)
def test_custom_multiheaded_model_list(v2_input, expected):
    @model(
        name="multi-headed",
        platform=ModelFramework.Custom,
    )
    def custom_multiheaded_model_list(a: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
        return [a, b]

    response = custom_multiheaded_model_list.request(v2_input)
    assert response == expected


def test_class_func_class():
    @model(
        name="classifier",
        platform=ModelFramework.TempoPipeline,
    )
    class MyCustomModel:
        @predictmethod
        def predict(self, X: str) -> str:
            return X

    x = MyCustomModel()

    r = x.predict("hello")
    assert r == "hello"
    r = x("hello")
    assert r == "hello"


def test_custom_loadmethod(custom_model):
    pred = custom_model(payload=np.array([1, 2, 3]))
    assert pred == np.array([6])


def test_model_save(custom_model: Model):
    custom_model.save(save_env=False)
    loaded = Model.load(custom_model.details.local_folder)

    assert len(custom_model.context.__dict__) > 0
    assert len(loaded.context.__dict__) == 0
