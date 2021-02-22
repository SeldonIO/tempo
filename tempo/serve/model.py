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
            local_folder=local_folder,
            uri=uri,
            platform=platform,
            inputs=inputs,
            outputs=outputs,
            runtime=runtime,
        )

    def __get__(self, instance, owner):
        if instance is None:
            return self  # Accessed from class, return unchanged

        return types.MethodType(self, instance)

    def __call__(self, *args, **kwargs) -> Any:
        if self._user_func is not None:
            return self._user_func(*args, **kwargs)
        else:
            return self.runtime.remote(self.details, *args, **kwargs)