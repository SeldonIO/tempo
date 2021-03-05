from tempo.kfserving.protocol import KFServingV2Protocol

def test_v2_from_any():
    data = "abc"
    p = KFServingV2Protocol()
    d = KFServingV2Protocol.create_v2_from_any(data,"a")
    assert d["name"] == "a"
    assert d["shape"][0] == 3
    assert d["data"] == [97,98,99]
    assert d["datatype"] == "BYTES"
