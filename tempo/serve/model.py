import types
import requests

from typing import Any, Type, List, Callable, Dict, Optional

from tempo.utils import logger
from tempo.serve.metadata import ModelDetails
from tempo.serve.runtime import Runtime
from tempo.serve.metadata import ModelFramework
from tempo.serve.base import BaseModel
from tempo.serve.constants import ModelDataType

class Model(BaseModel):
    def __init__(self, name: str,
                 runtime: Runtime = None,
                 local_folder: str = None,
                 uri: str = None,
                 platform: ModelFramework = None,
                 inputs: ModelDataType = None,
                 outputs: ModelDataType = None,
                 model_func: Callable[[Any], Any] = None,
                 ):
        super().__init__(name, model_func, runtime, inputs, outputs)
        self._runtime = runtime
        self._model_func = model_func

        self._details = ModelDetails(name=name, local_folder=local_folder, uri=uri, platform=platform, inputs=inputs, outputs=outputs)

        # Cached / mocked prediction
        self._prediction = None

    def __get__(self, instance, owner):
        if instance is None:
            return self  # Accessed from class, return unchanged

        return types.MethodType(self, instance)

    def _predict(self, req: dict) -> dict:
        endpoint = self._runtime.get_endpoint(self._details)
        response_raw = requests.post(endpoint, json=req)

        return response_raw.json()

    def set_prediction(self, prediction):
        self._prediction = prediction

    def __call__(self, *args, **kwargs) -> Any:
        if self._model_func is not None:
            return self._model_func(*args, **kwargs)
        else:
            endpoint = self._runtime.get_endpoint(self._details)
            protocol = self._runtime.get_protocol()

            logger.debug(f"Call {self._details.name} {endpoint}")

            if self._prediction is not None:
                return self._prediction

            req = protocol.to_protocol_request(*args, **kwargs)
            res = self._predict(req)

            return protocol.from_protocol_response(res, self.outputs)


    def deploy(self):
        self._runtime.deploy(self._details)

    def undeploy(self):
        self._runtime.undeploy(self._details)
