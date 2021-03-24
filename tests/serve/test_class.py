import numpy as np
import pytest


@pytest.mark.parametrize(
    "x_input, expected",
    [(np.array([[0.4, 2, 3, 4]]), [9.4])],
)
def test_class2(inference_pipeline_class, x_input, expected):
    y_pred = inference_pipeline_class.pipeline(x_input)
    assert y_pred == expected
    y_pred = inference_pipeline_class.pipeline(x_input)
    assert y_pred == expected
    assert inference_pipeline_class.get_counter() == 2
