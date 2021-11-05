from ast import literal_eval
from typing import Any, Dict, Optional, Type

import numpy as np

from tempo.serve.metadata import ModelDataArgs, ModelDetails
from tempo.serve.protocol import Protocol

_REQUEST_NUMPY_CONTENT_TYPE = {"content_type": "np"}

_v2tymap: Dict[str, np.dtype] = {
    "BOOL": np.dtype("bool"),
    "UINT8": np.dtype("uint8"),
    "UINT16": np.dtype("uint16"),
    "UINT32": np.dtype("uint32"),
    "UINT64": np.dtype("uint64"),
    "INT8": np.dtype("int8"),
    "INT16": np.dtype("int16"),
    "INT32": np.dtype("int32"),
    "INT64": np.dtype("int64"),
    "FP16": np.dtype("float32"),
    "FP32": np.dtype("float32"),
    "FP64": np.dtype("float64"),
}

_nptymap = dict([(value, key) for key, value in _v2tymap.items()])
_nptymap[np.dtype("float32")] = "FP32"  # Ensure correct mapping for ambiguous type


class V2Protocol(Protocol):
    @staticmethod
    def create_v2_from_np(arr: np.ndarray, name: str) -> Dict:
        if arr.dtype in _nptymap:
            return {
                "name": name,
                "datatype": _nptymap[arr.dtype],
                "data": arr.flatten().tolist(),
                "shape": list(arr.shape),
            }
        else:
            raise ValueError(f"Unknown numpy type {arr.dtype}")

    @staticmethod
    def create_v2_from_any(data: Any, name: str) -> Dict:
        if isinstance(data, str):
            b = list(bytes(data, "utf-8"))
        else:
            b = list(bytes(repr(data), "utf-8"))
        return {
            "name": name,
            "datatype": "BYTES",
            "data": b,
            "shape": [len(b)],
        }

    @staticmethod
    def convert_from_bytes(output: dict, ty: Optional[Type]) -> Any:
        if ty == str:
            return bytearray(output["data"]).decode("UTF-8")
        else:
            py_str = bytearray(output["data"]).decode("UTF-8")
            return literal_eval(py_str)

    @staticmethod
    def create_np_from_v2(data: list, ty: str, shape: list) -> np.ndarray:
        if ty in _v2tymap:
            npty = _v2tymap[ty]
            arr = np.array(data, dtype=npty)
            arr.shape = tuple(shape)
            return arr
        else:
            raise ValueError(f"V2 unknown type or type that can't be coerced {ty}")

    def get_predict_path(self, model_details: ModelDetails):
        return f"/v2/models/{model_details.name}/infer"

    def get_status_path(self, model_details: ModelDetails) -> str:
        return f"/v2/models/{model_details.name}/ready"

    def to_protocol_request(self, *args, **kwargs) -> Dict:
        # if len(args) > 0:
        #    raise ValueError("KFserving V2 protocol only supports named arguments")

        # We have numpy codec in mlserver for the inference request that would assume that there is only one numpy
        # array in the input to be parsed. this is a special case and we need to deal with it here as well.
        # So if we only have one input that is ndarray, we append `_REQUEST_NUMPY_CONTENT_TYPE`
        # to the inference request.

        inputs = []
        args_num = 0
        numpy_args_num = 0
        if len(args) > 0:
            for idx, raw in enumerate(args):
                raw_type = type(raw)
                args_num += 1
                if raw_type == np.ndarray:
                    numpy_args_num += 1
                    inputs.append(V2Protocol.create_v2_from_np(raw, "input-" + str(idx)))
                else:
                    inputs.append(V2Protocol.create_v2_from_any(raw, "input-" + str(idx)))
        else:
            for (name, raw) in kwargs.items():
                raw_type = type(raw)
                args_num += 1
                if raw_type == np.ndarray:
                    numpy_args_num += 1
                    inputs.append(V2Protocol.create_v2_from_np(raw, name))
                else:
                    inputs.append(V2Protocol.create_v2_from_any(raw, name))

        request_ret = {"inputs": inputs}
        np_inference_request_enabled = {"parameters": _REQUEST_NUMPY_CONTENT_TYPE}
        if args_num == numpy_args_num == 1:
            return {**request_ret, **np_inference_request_enabled}

        return request_ret

    @staticmethod
    def get_ty(name: str, idx: int, tys: ModelDataArgs) -> Optional[Type]:
        ty = tys[name]
        if ty is None:
            ty = tys[idx]
        if ty is None:
            return np.ndarray
        return ty

    def to_protocol_response(self, model_details: ModelDetails, *args, **kwargs) -> Dict:
        outputs = []
        for idx, raw in enumerate(args):
            raw_type = type(raw)

            if raw_type == np.ndarray:
                outputs.append(V2Protocol.create_v2_from_np(raw, "output" + str(idx)))
            else:
                outputs.append(V2Protocol.create_v2_from_any(raw, "output" + str(idx)))
        for name, raw in kwargs.items():
            raw_type = type(raw)

            if raw_type == np.ndarray:
                shape = list(raw.shape)
                data = raw.flatten().tolist()
                outputs.append({"name": name, "datatype": "FP32", "shape": shape, "data": data})
            else:
                outputs.append(V2Protocol.create_v2_from_any(raw, name))
        return {"model_name": model_details.name, "outputs": outputs}

    def from_protocol_request(self, res: Dict, tys: ModelDataArgs) -> Any:
        inp = {}
        for idx, input in enumerate(res["inputs"]):
            ty = V2Protocol.get_ty(input["name"], idx, tys)

            if input["datatype"] == "BYTES":
                inp[input["name"]] = V2Protocol.convert_from_bytes(input, ty)
            elif ty == np.ndarray:
                arr = V2Protocol.create_np_from_v2(input["data"], input["datatype"], input["shape"])
                inp[input["name"]] = arr
            else:
                raise ValueError(f"Unknown ty {ty} in conversion")

        if len(inp) == 1:
            return list(inp.values())[0]
        else:
            return inp

    def from_protocol_response(self, res: Dict, tys: ModelDataArgs) -> Any:
        out = {}
        for idx, output in enumerate(res["outputs"]):
            ty = V2Protocol.get_ty(output["name"], idx, tys)

            if output["datatype"] == "BYTES":
                out[output["name"]] = V2Protocol.convert_from_bytes(output, ty)
            elif ty == np.ndarray:
                arr = V2Protocol.create_np_from_v2(output["data"], output["datatype"], output["shape"])
                out[output["name"]] = arr
            else:
                raise ValueError(f"Unknown ty {ty} in conversion")

        if len(out) == 1:
            return list(out.values())[0]
        else:
            return out
