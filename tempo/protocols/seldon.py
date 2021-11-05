from typing import Any, Dict, Type, Union

import numpy as np

from tempo.serve.metadata import ModelDataArgs, ModelDetails
from tempo.serve.protocol import Protocol


class SeldonProtocol(Protocol):
    def get_predict_path(self, model_details: ModelDetails):
        # TODO: K8s backend needs to add namespace and model name
        #  return f"/seldon/{model_details.namespace}/{model_details.name}/api/v1.0/predictions"
        return "/api/v1.0/predictions"

    def get_status_path(self, model_details: ModelDetails) -> str:
        return "/api/v1.0/health/status"

    def to_protocol_request(self, *args, **kwargs) -> Dict:
        if not len(args) + len(kwargs) == 1:
            raise ValueError("Seldon protocol can only take a single input")

        if len(args) == 1:
            raw = args[0]
        else:
            raw = list(kwargs.values())[0]

        raw_type = type(raw)

        if raw_type == dict:
            return raw
        elif raw_type == list:
            return {"data": {"ndarray": raw}}
        elif raw_type == np.ndarray:
            return {"data": {"ndarray": raw.tolist()}}

        raise ValueError(f"Unknown input type {raw_type}")

    def to_protocol_response(self, model_details: ModelDetails, *args, **kwargs) -> Dict:
        return self.to_protocol_request(*args, **kwargs)

    def from_protocol_request(self, res: dict, tys: ModelDataArgs) -> Union[dict, np.ndarray]:
        if len(tys) > 1:
            raise ValueError("Seldon protocol can only return a single type")

        if tys[0] == Dict:
            # Return as-is
            return res

        ty: Type = np.ndarray
        if tys and tys[0] is not None:
            ty = tys[0]

        if ty == np.ndarray:
            if "data" in res:
                datadef = res["data"]
                if "tensor" in datadef:
                    tensor = datadef["tensor"]
                    return np.array(tensor["values"]).reshape(tensor["shape"])
                elif "ndarray" in datadef:
                    return np.array(datadef["ndarray"])
                # elif "tftensor" in datadef:
                #    tf_proto = TensorProto()
                #    json_format.ParseDict(datadef["tftensor"], tf_proto)
                #    return tf.make_ndarray(tf_proto)

            raise ValueError(f"Unknown data structure {res}")

        raise ValueError(f"unknown type {ty}")

    def from_protocol_response(self, res: Dict, tys: ModelDataArgs) -> Any:
        return self.from_protocol_request(res, tys)
