from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from tempo.protocols.v2 import V2Protocol
from tempo.serve.metadata import ModelDataArgs, ModelDetails
from tempo.serve.protocol import Protocol


class TensorflowProtocol(Protocol):
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
                    inputs.append(TensorflowProtocol.create_v1_from_np(raw))
        else:
            for (name, raw) in kwargs.items():
                raw_type = type(raw)

                if raw_type == np.ndarray:
                    inputs.append(TensorflowProtocol.create_v1_from_np(raw, name))
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
                    outputs.append(TensorflowProtocol.create_v1_from_np(raw))
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
            ty = TensorflowProtocol.get_ty(input["name"], idx, tys)

            if ty == np.ndarray:
                arr = V2Protocol.create_np_from_v2(input["data"], input["datatype"], input["shape"])
                inp[input["name"]] = arr
            else:
                raise ValueError(f"Unknown ty {ty} in conversion")

        if len(inp) == 1:
            return list(inp.values())[0]
        else:
            return inp

    def from_protocol_response(self, res: Dict, tys: ModelDataArgs) -> Any:
        if len(tys) <= 1:
            ty = TensorflowProtocol.get_ty(None, 0, tys)

            if ty == np.ndarray:
                return TensorflowProtocol.create_np_from_v1(res["predictions"])
            else:
                raise ValueError(f"Unknown ty {ty} in conversion")
        else:
            out = []
            for idx, output in enumerate(res["predictions"]):
                if type(output) == list:
                    for idx2, it in enumerate(output):
                        ty = TensorflowProtocol.get_ty(None, idx, tys)

                        if ty == np.ndarray:
                            arr = TensorflowProtocol.create_np_from_v1(it)
                            out.append(arr)
                        else:
                            raise ValueError(f"Unknown ty {ty} in conversion")

            if len(out) == 1:
                return out[0]
            else:
                return out
