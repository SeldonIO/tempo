import types
import requests

from typing import Any, Type, Optional, List

from tempo.utils import logger
from tempo.serve.metadata import ModelDetails
from tempo.serve.runtime import Runtime
from tempo.serve.metadata import ModelFramework, MetadataTensor


class Model:
    def __init__(self, name: str,
                 runtime: Runtime = None,
                 local_folder: str = None,
                 uri: str = None,
                 platform: ModelFramework = None,
                 inputs: List[MetadataTensor] = None,
                 outputs: List[MetadataTensor] = None
                 ):
        self._runtime = runtime

        self._details = ModelDetails(name=name, local_folder=local_folder, uri=uri, platform=platform, outputs=outputs)

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
        endpoint = self._runtime.get_endpoint(self._details)
        protocol = self._runtime.get_protocol()

        logger.debug(f"Call {self._details.name} {endpoint}")

        if self._prediction is not None:
            return self._prediction

        req = protocol.to_protocol_request(*args, **kwargs)
        res = self._predict(req)

        expected_return_types = self._get_output_types()
        return protocol.from_protocol_response(res, expected_return_types)

    def _get_output_types(self) -> List[Type]:
        if not self._details.outputs:
            return []

        tys = []
        for output_meta in self._details.outputs:
            parameters = output_meta.parameters
            if not parameters:
                tys.append(None)
            else:
                tys.append(parameters.ext_datatype)
        return tys

    def deploy(self):
        self._runtime.deploy(self._details)

    def undeploy(self):
        self._runtime.undeploy(self._details)
