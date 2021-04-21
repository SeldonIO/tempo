from typing import Any, Callable

from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.base import BaseModel
from tempo.serve.metadata import ModelFramework, RuntimeOptions
from tempo.serve.protocol import Protocol
from tempo.serve.types import ModelDataType


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
        """

        Parameters
        ----------
        name
         Name of the pipeline. Needs to be Kubernetes compliant.
        protocol
         :class:`tempo.serve.protocol.Protocol`. Defaults to KFserving V2.
        local_folder
         Location of local artifacts.
        uri
         Location of remote artifacts.
        platform
         The :class:`tempo.serve.metadata.ModelFramework`
        inputs
         The input types.
        outputs
         The output types.
        conda_env
         The conda environment name to use. If not specified will look for conda.yaml in
         local_folder or generate from current running environment.
        runtime_options
         The runtime options. Can be left empty and set when creating a runtime.

        """
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
