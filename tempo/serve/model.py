from typing import Any, Callable

from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.base import BaseModel
from tempo.serve.constants import ModelDataType
from tempo.serve.metadata import ModelFramework, RuntimeOptions
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
        runtime_options: RuntimeOptions = RuntimeOptions(),
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
            protocol=protocol,
            runtime_options=runtime_options,
        )
