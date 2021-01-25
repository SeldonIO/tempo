import numpy as np
from typing import Union, Type, Optional, Dict, List, Any
from tempo.serve.protocol import Protocol

_v2tymap = {
    "BOOL":np.dtype('bool'),
    "UINT8": np.dtype('uint8'),
    "UINT16": np.dtype('uint16'),
    "UINT32": np.dtype('uint32'),
    "UINT64": np.dtype('uint64'),
    "INT8": np.dtype('int8'),
    "INT16": np.dtype('int16'),
    "INT32": np.dtype('int32'),
    "INT64": np.dtype('int64'),
    "FP16": np.dtype('float32'),
    "FP32": np.dtype('float32'),
    "FP64": np.dtype('float64')
}

_nptymap = dict([reversed(i) for i in _v2tymap.items()])
_nptymap[np.dtype('float32')] = "FP32" #Ensure correct mapping for ambiguous type


class KFServingV2Protocol(Protocol):

    def __init__(self, model_name:str):
        self.model_name = model_name

    @staticmethod
    def create_v2_from_np(arr: np.ndarray, name: str) -> Dict:
        if arr.dtype in _nptymap:
            return { "name": name, "datatype":_nptymap[arr.dtype], "data": arr.flatten().tolist(), "shape":list(arr.shape)}
        else:
            raise ValueError(f"Unknown numpy type {arr.dtype}")

    @staticmethod
    def create_np_from_v2(data: list, ty: str, shape: list) -> np.array:
        if ty in _v2tymap:
            npty = _v2tymap[ty]
            arr = np.array(data, dtype=npty)
            arr.shape = tuple(shape)
            return arr
        else:
            raise ValueError(f"V2 unknown type or type that can't be coerced {ty}")

    def get_predict_path(self):
        return f"/v2/models/{self.model_name}/infer"

    def to_protocol_request(self, *args, **kwargs) -> Dict:
        if len(args) > 0:
            raise ValueError("KFserving V2 protocol only supports named arguments")

        inputs = []
        for (name,raw) in kwargs.items():
            raw_type = type(raw)

            if raw_type == np.ndarray:
                inputs.append(KFServingV2Protocol.create_v2_from_np(raw, name))
            else:
                raise ValueError(f"Unknown input type {raw_type}")

    def to_protocol_response(self, *args, **kwargs) -> Dict:
        outputs = []
        for idx,raw in enumerate(args):
            raw_type = type(raw)

            if raw_type == np.ndarray:
                outputs.append(KFServingV2Protocol.create_v2_from_np(raw,"output"+str(idx)))
            else:
                raise ValueError(f"Unknown input type {raw_type}")
        for name,raw in kwargs.items():
            raw_type = type(raw)

            if raw_type == np.ndarray:
                shape = list(raw.shape)
                data = raw.flatten().tolist()
                outputs.append(
                    {"name": name, "datatype": "FP32", "shape": shape, "data": data})
            else:
                raise ValueError(f"Unknown input type {raw_type}")
        return {"model_name":self.model_name, "outputs":outputs}

    def from_protocol_request(self, res: dict, tys: List[Type]) -> Any:
        inp = {}
        for idx, input in enumerate(res["inputs"]):
            if tys is not None and len(tys) > idx:
                ty = tys[idx]
            else:
                ty = np.ndarray
            if ty == np.ndarray:
                arr = KFServingV2Protocol.create_np_from_v2(input["data"], input["datatype"],
                                                            input["shape"])
                inp[input["name"]] = arr
            else:
                raise ValueError(f"Unknown ty {ty} in conversion")

        if len(inp) == 1:
            return list(inp.values())[0]
        else:
            return inp

    def from_protocol_response(self, res: Dict, tys: List[Type]) -> Any:
        out = {}
        for idx, output in enumerate(res["outputs"]):
            if tys is not None and len(tys) > idx:
                ty = tys[idx]
            else:
                ty = np.ndarray
            if ty == np.ndarray:
                arr = KFServingV2Protocol.create_np_from_v2(input["data"], input["datatype"],
                                                            input["shape"])
                out[output["name"]] = arr
            else:
                raise ValueError(f"Unknown ty {ty} in conversion")

        if len(out) == 1:
            return list(out.values())[0]
        else:
            return out


