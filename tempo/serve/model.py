import types
import requests

from typing import Any, Callable

from tempo.serve.runtime import Runtime
from tempo.serve.metadata import ModelFramework
from tempo.serve.base import BaseModel
from tempo.serve.constants import ModelDataType


class Model(BaseModel):
    def __init__(
        self,
        name: str,
        runtime: Runtime = None,
        local_folder: str = None,
        uri: str = None,
        platform: ModelFramework = None,
        inputs: ModelDataType = None,
        outputs: ModelDataType = None,
        model_func: Callable[[Any], Any] = None,
    ):
        super().__init__(
            name,
            # TODO: Should we unify names?
            user_func=model_func,
            # TODO: What should happen if runtime is None?
            protocol=runtime.get_protocol(),
            local_folder=local_folder,
            uri=uri,
            platform=platform,
            inputs=inputs,
            outputs=outputs,
        )

        self._runtime = runtime

    def __get__(self, instance, owner):
        if instance is None:
            return self  # Accessed from class, return unchanged

        return types.MethodType(self, instance)

    def deploy(self):
        self._runtime.deploy(self.details)

    def undeploy(self):
        self._runtime.undeploy(self.details)

    def to_k8s_yaml(self) -> str:
        """
        Get k8s yaml
        """
        return self._runtime.to_k8s_yaml(self.details)

    def set_runtime(self, runtime: Runtime):
        self._runtime = runtime
        self.protocol = runtime.get_protocol()

    def get_endpoint(self):
        return self._runtime.get_endpoint(self.details)

    def wait_ready(self, timeout_secs=None):
        return self._runtime.wait_ready(self.details, timeout_secs=timeout_secs)

    def remote(self, *args, **kwargs):
        return self._runtime.remote(self.details, *args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        if self._user_func is not None:
            return self._user_func(*args, **kwargs)
        else:
            return self._runtime.remote(self.details, *args, **kwargs)