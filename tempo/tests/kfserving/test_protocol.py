import pytest

from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.metadata import ModelDataArg, ModelDataArgs


@pytest.mark.parametrize(
    "data, expected",
    [("abc", [97, 98, 99]), ({"a": 1}, [123, 39, 97, 39, 58, 32, 49, 125])],
)
def test_v2_from_any(data, expected):
    d = KFServingV2Protocol.create_v2_from_any(data, "a")
    assert d["name"] == "a"
    assert d["data"] == expected
    assert d["datatype"] == "BYTES"


@pytest.mark.parametrize(
    "data, ty, expected",
    [([97, 98, 99], str, "abc"), ([123, 39, 97, 39, 58, 32, 49, 125], dict, {"a": 1})],
)
def test_convert_from_bytes(data, ty, expected):
    output = {"data": data}
    res = KFServingV2Protocol.convert_from_bytes(output, ty)
    assert res == expected


def test_v2_from_protocol_response():
    res = {"outputs": [{"name": "a", "data": [97, 98, 99], "datatype": "BYTES"}]}
    modelTyArgs = ModelDataArgs(args=[ModelDataArg(ty=str, name=None)])
    v2 = KFServingV2Protocol()
    res = v2.from_protocol_response(res, modelTyArgs)
    print(res)
