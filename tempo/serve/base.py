from typing import Callable, List, Any, Dict, Optional, Type, get_type_hints, Tuple
import typing

from tempo.serve.loader import load_custom, save_custom
from tempo.serve.protocol import Protocol
from tempo.serve.constants import ModelDataType
from tempo.serve.metadata import ModelDataArgs, ModelDataArg


class BaseModel:

    def __init__(self, name: str,
                 user_func: Callable[[Any], Any],
                 protocol: Protocol = None,
                 inputs: ModelDataType = None,
                 outputs: ModelDataType = None):
        self._name = name
        self._user_func = user_func
        self.protocol = protocol
        input_args = []
        output_args = []
        if inputs is None and outputs is None:
            if user_func is not None:
                hints = get_type_hints(user_func)
                for k,v in hints.items():
                    if k == "return":
                        if isinstance(v, typing._GenericAlias):
                            targs = v.__args__
                            for targ in targs:
                                output_args.append(ModelDataArg(ty=targ))
                        else:
                            output_args.append(ModelDataArg(ty=v))
                    else:
                        input_args.append(ModelDataArg(name=k,ty=v))
        else:
            if type(outputs) == Dict:
                for k,v in outputs.items():
                    output_args.append(ModelDataArg(name=k,ty=v))
            elif type(outputs) == Tuple:
                for ty in list(outputs):
                    output_args.append(ModelDataArg(ty=ty))
            else:
                output_args.append(ModelDataArg(ty=outputs))

            if type(inputs) == Dict:
                for k, v in inputs.items():
                    input_args.append(ModelDataArg(name=k, ty=v))
            elif type(inputs) == Tuple:
                for ty in list(inputs):
                    input_args.append(ModelDataArg(ty=ty))
            else:
                input_args.append(ModelDataArg(ty=inputs))
        self.inputs: ModelDataArgs = ModelDataArgs(args=input_args)
        self.outputs: ModelDataArgs = ModelDataArgs(args=output_args)
        self.cls = None

    def set_cls(self, cls):
        self.cls = cls

    @classmethod
    def load(cls, file_path: str):
        return load_custom(file_path)

    def save(self, file_path: str):
        save_custom(self, file_path)

    def request(self, req: Dict) -> Dict:
        req_converted = self.protocol.from_protocol_request(req, self.inputs)
        if type(req_converted) == dict:
            if self.cls is not None:
                response = self._user_func(self.cls, **req_converted)
            else:
                response = self._user_func(**req_converted)
        elif type(req_converted) == list or type(req_converted) == tuple:
            if self.cls is not None:
                response = self._user_func(self.cls, *req_converted)
            else:
                response = self._user_func(*req_converted)
        else:
            if self.cls is not None:
                response = self._user_func(self.cls, req_converted)
            else:
                response = self._user_func(req_converted)
        if type(response) == dict:
            response_converted = self.protocol.to_protocol_response(**response)
        elif type(response) == list or type(response) == tuple:
            response_converted = self.protocol.to_protocol_response(*response)
        else:
            response_converted = self.protocol.to_protocol_response(response)
        return response_converted

