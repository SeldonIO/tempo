from ast import literal_eval
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from tempo.serve.metadata import ModelDataArgs, ModelDetails
from tempo.serve.protocol import Protocol

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


class KFServingV2Protocol(Protocol):
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

        inputs = []
        if len(args) > 0:
            for idx, raw in enumerate(args):
                raw_type = type(raw)
                if raw_type == np.ndarray:
                    inputs.append(KFServingV2Protocol.create_v2_from_np(raw, "input-" + str(idx)))
                else:
                    inputs.append(KFServingV2Protocol.create_v2_from_any(raw, "input-" + str(idx)))
        else:
            for (name, raw) in kwargs.items():
                raw_type = type(raw)

                if raw_type == np.ndarray:
                    inputs.append(KFServingV2Protocol.create_v2_from_np(raw, name))
                else:
                    inputs.append(KFServingV2Protocol.create_v2_from_any(raw, name))

        return {"inputs": inputs}

    @staticmethod
    def get_ty(name: str, idx: int, tys: ModelDataArgs) -> Optional[Type]:
        ty = tys[name]
        if ty is None:
            ty = tys[idx]
        # if ty is None:
        #    return np.ndarray
        return ty

    def to_protocol_response(self, model_details: ModelDetails, *args, **kwargs) -> Dict:
        outputs = []
        for idx, raw in enumerate(args):
            raw_type = type(raw)

            if raw_type == np.ndarray:
                outputs.append(KFServingV2Protocol.create_v2_from_np(raw, "output" + str(idx)))
            else:
                outputs.append(KFServingV2Protocol.create_v2_from_any(raw, "output" + str(idx)))
        for name, raw in kwargs.items():
            raw_type = type(raw)

            if raw_type == np.ndarray:
                shape = list(raw.shape)
                data = raw.flatten().tolist()
                outputs.append({"name": name, "datatype": "FP32", "shape": shape, "data": data})
            else:
                outputs.append(KFServingV2Protocol.create_v2_from_any(raw, name))
        return {"model_name": model_details.name, "outputs": outputs}

    def from_protocol_request(self, res: Dict, tys: ModelDataArgs) -> Any:
        inp = {}
        for idx, input in enumerate(res["inputs"]):
            ty = KFServingV2Protocol.get_ty(input["name"], idx, tys)

            if ty == np.ndarray:
                arr = KFServingV2Protocol.create_np_from_v2(input["data"], input["datatype"], input["shape"])
                inp[input["name"]] = arr
            elif input["datatype"] == "BYTES":
                inp[input["name"]] = KFServingV2Protocol.convert_from_bytes(input, ty)
            else:
                raise ValueError(f"Unknown ty {ty} in conversion")

        if len(inp) == 1:
            return list(inp.values())[0]
        else:
            return inp

    def from_protocol_response(self, res: Dict, tys: ModelDataArgs) -> Any:
        out = {}
        for idx, output in enumerate(res["outputs"]):
            ty = KFServingV2Protocol.get_ty(output["name"], idx, tys)

            if ty == np.ndarray:
                arr = KFServingV2Protocol.create_np_from_v2(output["data"], output["datatype"], output["shape"])
                out[output["name"]] = arr
            elif output["datatype"] == "BYTES":
                out[output["name"]] = KFServingV2Protocol.convert_from_bytes(output, ty)
            else:
                raise ValueError(f"Unknown ty {ty} in conversion")

        if len(out) == 1:
            return list(out.values())[0]
        else:
            return out


class KFServingV1Protocol(Protocol):
    @staticmethod
    def create_v1_from_np(arr: np.ndarray, name: str = None) -> list:
        return arr.tolist()

    @staticmethod
    def create_np_from_v1(data: list) -> np.ndarray:
        arr = np.array(data)
        return arr

    def get_predict_path(self, model_details: ModelDetails):
        return f"/v1/models/{model_details.name}:predict"

    def get_status_path(self, model_details: ModelDetails) -> str:
        return f"/v1/models/{model_details.name}"

    def to_protocol_request(self, *args, **kwargs) -> Dict:
        if len(args) > 0 and len(kwargs.values()) > 0:
            raise ValueError("KFserving V1 protocol only supports either named or unamed arguments but not both")

        inputs = []
        if len(args) > 0:
            for raw in args:
                raw_type = type(raw)

                if raw_type == np.ndarray:
                    inputs.append(KFServingV1Protocol.create_v1_from_np(raw))
        else:
            for (name, raw) in kwargs.items():
                raw_type = type(raw)

                if raw_type == np.ndarray:
                    inputs.append(KFServingV1Protocol.create_v1_from_np(raw, name))
                else:
                    raise ValueError(f"Unknown input type {raw_type}")

        if len(inputs) == 1:
            return {"instances": inputs[0]}
        else:
            return {"instances": inputs}

    @staticmethod
    def get_ty(name: Optional[str], idx: int, tys: ModelDataArgs) -> Type:
        ty = None
        if name is not None:
            ty = tys[name]
        if ty is None:
            ty = tys[idx]
        if ty is None:
            return np.ndarray
        return ty

    def to_protocol_response(self, model_details: ModelDetails, *args, **kwargs) -> Dict:
        outputs: List[Union[Dict, List]] = []
        if len(args) > 0:
            for idx, raw in enumerate(args):
                raw_type = type(raw)

                if raw_type == np.ndarray:
                    outputs.append(KFServingV1Protocol.create_v1_from_np(raw))
                else:
                    raise ValueError(f"Unknown input type {raw_type}")
        else:
            for name, raw in kwargs.items():
                raw_type = type(raw)

                if raw_type == np.ndarray:
                    data = raw.tolist()
                    outputs.append({name: data})
                else:
                    raise ValueError(f"Unknown input type {raw_type}")
        return {"predictions": outputs}

    def from_protocol_request(self, res: Dict, tys: ModelDataArgs) -> Any:
        inp = {}
        for idx, input in enumerate(res["inputs"]):
            ty = KFServingV1Protocol.get_ty(input["name"], idx, tys)

            if ty == np.ndarray:
                arr = KFServingV2Protocol.create_np_from_v2(input["data"], input["datatype"], input["shape"])
                inp[input["name"]] = arr
            else:
                raise ValueError(f"Unknown ty {ty} in conversion")

        if len(inp) == 1:
            return list(inp.values())[0]
        else:
            return inp

    def from_protocol_response(self, res: Dict, tys: ModelDataArgs) -> Any:
        if len(tys) <= 1:
            ty = KFServingV1Protocol.get_ty(None, 0, tys)

            if ty == np.ndarray:
                return KFServingV1Protocol.create_np_from_v1(res["predictions"])
            else:
                raise ValueError(f"Unknown ty {ty} in conversion")
        else:
            out = []
            for idx, output in enumerate(res["predictions"]):
                if type(output) == list:
                    for idx2, it in enumerate(output):
                        ty = KFServingV1Protocol.get_ty(None, idx, tys)

                        if ty == np.ndarray:
                            arr = KFServingV1Protocol.create_np_from_v1(it)
                            out.append(arr)
                        else:
                            raise ValueError(f"Unknown ty {ty} in conversion")

            if len(out) == 1:
                return out[0]
            else:
                return out
