import types
import requests

from typing import Any, Type, List, Callable, Dict, Optional

from tempo.serve.metadata import ModelDetails
from tempo.serve.runtime import Runtime
from tempo.serve.metadata import ModelFramework
from tempo.serve.base import BaseModel
from tempo.serve.constants import ModelDataType
from tempo.serve.loader import download, upload


class Model(BaseModel):
    def __init__(self, name: str,
                 runtime: Runtime = None,
                 local_folder: str = None,
                 uri: str = None,
                 platform: ModelFramework = None,
                 inputs: ModelDataType = None,
                 outputs: ModelDataType = None,
                 model_func: Callable[[Any], Any] = None
                 ):
        super().__init__(name, model_func, runtime.get_protocol(), inputs, outputs)
        self._model_func = model_func
        self._runtime = runtime
        self._details = ModelDetails(name=name, local_folder=local_folder, uri=uri, platform=platform, inputs=inputs, outputs=outputs)

    def __get__(self, instance, owner):
        if instance is None:
            return self  # Accessed from class, return unchanged

        return types.MethodType(self, instance)

    def _predict(self, req: dict) -> dict:
        endpoint = self._runtime.get_endpoint(self._details)
        response_raw = requests.post(endpoint, json=req)
        return response_raw.json()

    def __call__(self, *args, **kwargs) -> Any:
        if self._model_func is not None:
            return self._model_func(*args, **kwargs)
        else:
            protocol = self._runtime.get_protocol()
            req = protocol.to_protocol_request(*args, **kwargs)
            res = self._predict(req)
            return protocol.from_protocol_response(res, self.outputs)


    def deploy(self):
        self._runtime.deploy(self._details)

    def undeploy(self):
        self._runtime.undeploy(self._details)

    def to_k8s_yaml(self) -> str:
        """
        Get k8s yaml
        """
        return self._runtime.to_k8s_yaml(self._details)

    def upload(self):
        """
        Upload from local folder to uri
        """
        upload(self._details.local_folder, self._details.uri)

    def download(self):
        """
        Download from uri to local folder
        """
        download(self._details.uri, self._details.local_folder)

    def set_runtime(self, runtime: Runtime):
        self._runtime = runtime
        self.protocol = runtime.get_protocol()

    def get_endpoint(self):
        return self._runtime.get_endpoint(self._details)

    def wait_ready(self, timeout_secs=None):
        return self._runtime.wait_ready(self._details,timeout_secs=timeout_secs)