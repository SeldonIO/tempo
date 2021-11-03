import numpy as np
import pytest

from tempo.protocols.v2 import _REQUEST_NUMPY_CONTENT_TYPE, V2Protocol
from tempo.serve.metadata import ModelDataArg, ModelDataArgs


@pytest.mark.parametrize(
    "data, expected",
    [("abc", [97, 98, 99]), ({"a": 1}, [123, 39, 97, 39, 58, 32, 49, 125])],
)
def test_v2_from_any(data, expected):
    d = V2Protocol.create_v2_from_any(data, "a")
    assert d["name"] == "a"
    assert d["data"] == expected
    assert d["datatype"] == "BYTES"


@pytest.mark.parametrize(
    "data, ty, expected",
    [([97, 98, 99], str, "abc"), ([123, 39, 97, 39, 58, 32, 49, 125], dict, {"a": 1})],
)
def test_convert_from_bytes(data, ty, expected):
    output = {"data": data}
    res = V2Protocol.convert_from_bytes(output, ty)
    assert res == expected


def test_v2_from_protocol_response():
    res = {"outputs": [{"name": "a", "data": [97, 98, 99], "datatype": "BYTES"}]}
    modelTyArgs = ModelDataArgs(args=[ModelDataArg(ty=str, name=None)])
    v2 = V2Protocol()
    res = v2.from_protocol_response(res, modelTyArgs)


def test_v2_to_protocol_request_numpy():
    v2 = V2Protocol()
    data = np.random.randn(1, 28 * 28)
    request = v2.to_protocol_request(data)
    expected_request = {
        "parameters": _REQUEST_NUMPY_CONTENT_TYPE,
        "inputs": [{"name": "input-0", "datatype": "FP64", "data": data.flatten().tolist(), "shape": list(data.shape)}],
    }

    assert expected_request == request


def test_v2_to_protocol_request_other():
    v2 = V2Protocol()
    data = 1
    request = v2.to_protocol_request(data)
    # we should not have the "parameters", mainly so that content_type= "np" is not present.
    # this seems a bit convoluted, so we need to find a better way perhaps for dealing with inference types in tempo
    assert "parameters" not in request
