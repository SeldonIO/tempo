from typing import Any, Callable

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
        conda_env: str = None,
        deployed: bool = False,
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
            conda_env=conda_env,
            deployed=deployed,
        )

    def __call__(self, *args, **kwargs) -> Any:
        if self._user_func is not None:
            if self.deployed:
                return self.remote(*args, **kwargs)
            else:
                if self.cls is not None:
                    return self._user_func(self.cls, *args, **kwargs)
                else:
                    return self._user_func(*args, **kwargs)

        return self.remote(*args, **kwargs)
