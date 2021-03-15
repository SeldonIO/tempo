import types
from typing import Any, Callable

from tempo.errors import UndefinedRuntime
from tempo.serve.base import BaseModel
from tempo.serve.constants import ModelDataType
from tempo.serve.metadata import ModelFramework
from tempo.serve.runtime import Runtime


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
        model_func: Callable[..., Any] = None,
        conda_env: str = None
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
            conda_env=conda_env
        )

    def __call__(self, *args, **kwargs) -> Any:
        if self._user_func is not None:
            if not self.cls is None:
                return self._user_func(self.cls, *args, **kwargs)
            else:
                return self._user_func(*args, **kwargs)

        if self.runtime is None:
            raise UndefinedRuntime(self.details.name)

        return self.runtime.remote(self.details, *args, **kwargs)
