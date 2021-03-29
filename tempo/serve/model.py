from typing import Any, Callable

from tempo.serve.base import BaseModel
from tempo.serve.runtime import Runtime
from tempo.serve.metadata import ModelFramework
from tempo.serve.constants import ModelDataType
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.protocol import Protocol


class Model(BaseModel):
    def __init__(
        self,
        name: str,
        protocol: Protocol = KFServingV2Protocol(),
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
            conda_env=conda_env,
            deployed=deployed,
            protocol=protocol,
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
