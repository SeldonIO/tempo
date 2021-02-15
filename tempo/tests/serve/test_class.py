from tempo.serve.utils import pipeline, predictmethod
from tempo.serve.model import Model
from tempo.seldon.docker import SeldonDockerRuntime
import numpy as np
import time
import pytest


@pytest.mark.parametrize(
    "x_input, expected",
    [
        (
            {
                "inputs": [
                    {
                        "name": "input0",
                        "datatype": "FP64",
                        "shape": [1, 4],
                        "data": [0.4, 2, 3, 4],
                    }
                ]
            },
            {
                "model_name": "mypipeline",
                "outputs": [
                    {"name": "output0", "datatype": "FP64", "shape": [1], "data": [2.0]}
                ],
            },
        )
    ],
)
def test_class2(inference_pipeline_v3, x_input, expected):

    y_pred = inference_pipeline_v3.pipeline.request(x_input)
    assert y_pred == expected
    y_pred = inference_pipeline_v3.pipeline.request(x_input)
    assert y_pred == expected
    assert inference_pipeline_v3.get_counter() == 2
