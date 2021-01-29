from tempo.serve.utils import model
from tempo.serve.metadata import (
    ModelFramework,
)
from typing import Tuple, List
import pytest
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.kfserving.protocol import KFServingV2Protocol

import numpy as np

#
# Test single input output model with type annotations
#
@pytest.mark.parametrize(
    "v2_input, expected",
    [
        ({"inputs":[{"name":"a","datatype":"FP64","shape":[1,4],"data":[1,2,3,4]}]},
         {'model_name': 'custom', 'outputs': [{'name': 'output0', 'datatype': 'FP64', 'data': [1.0, 2.0, 3.0, 4.0], 'shape': [1, 4]}]})
    ]
)
def test_custom_model(v2_input,expected):
    @model(name="test-iris-sklearn",
           runtime=SeldonDockerRuntime(protocol=KFServingV2Protocol(model_name="custom")),
           platform=ModelFramework.Custom,
           uri="gs://seldon-models/custom",
           local_folder="custom_iris_path"
           )
    def custom_model(a: np.ndarray) -> np.ndarray:
        return a

    response = custom_model.request(v2_input)
    assert response == expected


#
# Test single input output model with type annotations
#

@pytest.mark.parametrize(
    "v2_input, expected",
    [
        ({"inputs":[{"name":"a","datatype":"FP64","shape":[1,4],"data":[1,2,3,4]}]},
         {'model_name': 'custom', 'outputs': [{'name': 'output0', 'datatype': 'FP64', 'data': [1.0, 2.0, 3.0, 4.0], 'shape': [1, 4]}]})
    ]
)
def test_custom_model_decorator_types(v2_input,expected):
    @model(name="test-iris-sklearn",
           runtime=SeldonDockerRuntime(protocol=KFServingV2Protocol(model_name="custom")),
           platform=ModelFramework.Custom,
           uri="gs://seldon-models/custom",
           local_folder="custom_iris_path",
           inputs=np.ndarray,
           outputs=np.ndarray
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
        ({"inputs":[{"name":"a","datatype":"FP64","shape":[1,4],"data":[1,2,3,4]},
                                {"name":"b","datatype":"FP64","shape":[1,2],"data":[4,5]}]},
         {'model_name': 'multi-headed', 'outputs': [{'name': 'output0', 'datatype': 'FP64', 'data': [1.0, 2.0, 3.0, 4.0], 'shape': [1, 4]},
                                                    {'name': 'output1', 'datatype': 'FP64', 'data': [4.0, 5.0], 'shape': [1, 2]}]})
    ]
)
def test_custom_multiheaded_model_tuple(v2_input,expected):
    @model(name="test-iris-sklearn",
           runtime=SeldonDockerRuntime(protocol=KFServingV2Protocol(model_name="multi-headed")),
           platform=ModelFramework.Custom,
           uri="gs://seldon-models/custom",
           local_folder="custom_iris_path"
           )
    def custom_multiheaded_model_tuple(a: np.ndarray, b: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        return a, b

    response = custom_multiheaded_model_tuple.request(v2_input)
    assert response == expected


#
# Test multi-headed input output model with type annotations with list return
#
@pytest.mark.parametrize(
    "v2_input, expected",
    [
        ({"inputs":[{"name":"a","datatype":"FP64","shape":[1,4],"data":[1,2,3,4]},
                                {"name":"b","datatype":"FP64","shape":[1,2],"data":[4,5]}]},
         {'model_name': 'multi-headed', 'outputs': [{'name': 'output0', 'datatype': 'FP64', 'data': [1.0, 2.0, 3.0, 4.0], 'shape': [1, 4]},
                                                    {'name': 'output1', 'datatype': 'FP64', 'data': [4.0, 5.0], 'shape': [1, 2]}]})
    ]
)
def test_custom_multiheaded_model_list(v2_input,expected):
    @model(name="test-iris-sklearn",
           runtime=SeldonDockerRuntime(protocol=KFServingV2Protocol(model_name="multi-headed")),
           platform=ModelFramework.Custom,
           uri="gs://seldon-models/custom",
           local_folder="custom_iris_path"
           )
    def custom_multiheaded_model_list(a: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
        return a, b
    response = custom_multiheaded_model_list.request(v2_input)
    assert response == expected