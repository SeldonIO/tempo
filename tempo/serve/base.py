from __future__ import annotations

import os
import tempfile
from os import path
from pydoc import locate
from typing import Any, Callable, Dict, Optional, Tuple, get_type_hints

from tempo.errors import UndefinedCustomImplementation
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.constants import (
    ENV_K8S_SERVICE_HOST,
    DefaultCondaFile,
    DefaultEnvFilename,
    DefaultModelFilename,
    ModelDataType,
)
from tempo.serve.loader import load_custom, save_custom, save_environment
from tempo.serve.metadata import ModelDataArg, ModelDataArgs, ModelDetails, ModelFramework, RuntimeOptions
from tempo.serve.protocol import Protocol
from tempo.serve.remote import Remote
from tempo.serve.runtime import ModelSpec, Runtime
from tempo.utils import logger, tempo_settings


class BaseModel:
    def __init__(
        self,
        name: str,
        user_func: Callable[..., Any] = None,
        local_folder: str = None,
        uri: str = None,
        platform: ModelFramework = None,
        inputs: ModelDataType = None,
        outputs: ModelDataType = None,
        conda_env: str = None,
        protocol: Protocol = None,
        runtime_options: RuntimeOptions = RuntimeOptions(),
    ):
        if protocol is None:
            protocol = KFServingV2Protocol()
        self._name = name
        self._user_func = user_func
        self.conda_env_name = conda_env

        if uri is None:
            uri = ""

        local_folder = self._get_local_folder(local_folder)
        input_args, output_args = self._get_args(inputs, outputs)

        self.details = ModelDetails(
            name=name,
            local_folder=local_folder,
            uri=uri,
            platform=platform,
            inputs=input_args,
            outputs=output_args,
        )

        self.cls = None
        self.protocol = protocol
        self.model_spec = ModelSpec(model_details=self.details, protocol=self.protocol, runtime_options=runtime_options)

        self.use_remote: bool = False

    def set_remote(self, val: bool):
        self.use_remote = val

    def _get_args(
        self, inputs: ModelDataType = None, outputs: ModelDataType = None
    ) -> Tuple[ModelDataArgs, ModelDataArgs]:
        input_args = []
        output_args = []

        if isinstance(inputs, ModelDataArgs) and isinstance(outputs, ModelDataArgs):
            return inputs, outputs
        elif inputs is None and outputs is None:
            if self._user_func is not None:
                hints = get_type_hints(self._user_func)
                for k, v in hints.items():
                    if k == "return":
                        if hasattr(v, "__args__"):
                            # NOTE: If `__args__` are present, assume this as a
                            # `typing.Generic`, like `Tuple`
                            targs = v.__args__
                            for targ in targs:
                                output_args.append(ModelDataArg(ty=targ))
                        else:
                            output_args.append(ModelDataArg(ty=v))
                    else:
                        input_args.append(ModelDataArg(name=k, ty=v))
        else:
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    output_args.append(ModelDataArg(name=k, ty=v))
            elif isinstance(outputs, tuple):
                for ty in list(outputs):
                    output_args.append(ModelDataArg(ty=ty))
            else:
                output_args.append(ModelDataArg(ty=outputs))

            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    input_args.append(ModelDataArg(name=k, ty=v))
            elif isinstance(inputs, tuple):
                for ty in list(inputs):
                    input_args.append(ModelDataArg(ty=ty))
            else:
                input_args.append(ModelDataArg(ty=inputs))

        return ModelDataArgs(args=input_args), ModelDataArgs(args=output_args)

    def _get_local_folder(self, local_folder: str = None) -> Optional[str]:
        if not local_folder:
            # TODO: Should we do cleanup at some point?
            local_folder = tempfile.mkdtemp()

        return local_folder

    def set_cls(self, cls):
        self.cls = cls

    @classmethod
    def load(cls, folder: str) -> "BaseModel":
        file_path_pkl = os.path.join(folder, DefaultModelFilename)
        return load_custom(file_path_pkl)

    def save(self, save_env=True):
        logger.info("Saving environment")
        if not self._user_func:
            # Nothing to save
            return

        file_path_pkl = os.path.join(self.details.local_folder, DefaultModelFilename)
        logger.info("Saving tempo model to %s", file_path_pkl)
        save_custom(self, file_path_pkl)

        if save_env:
            file_path_env = os.path.join(self.details.local_folder, DefaultEnvFilename)
            conda_env_file_path = path.join(self.details.local_folder, DefaultCondaFile)
            if not path.exists(conda_env_file_path):
                conda_env_file_path = None

            save_environment(
                conda_pack_file_path=file_path_env,
                conda_env_file_path=conda_env_file_path,
                env_name=self.conda_env_name,
            )

    def request(self, req: Dict) -> Dict:

        if self._user_func is None:
            raise UndefinedCustomImplementation(self.details.name)

        req_converted = self.protocol.from_protocol_request(req, self.details.inputs)
        if type(req_converted) == dict:
            if self.cls is not None:
                response = self._user_func(self.cls, **req_converted)
            else:
                response = self._user_func(**req_converted)
        elif type(req_converted) == list or type(req_converted) == tuple:
            if self.cls is not None:
                response = self._user_func(self.cls, *req_converted)
            else:
                response = self._user_func(*req_converted)
        else:
            if self.cls is not None:
                response = self._user_func(self.cls, req_converted)
            else:
                response = self._user_func(req_converted)

        if type(response) == dict:
            response_converted = self.protocol.to_protocol_response(self.details, **response)
        elif type(response) == list or type(response) == tuple:
            response_converted = self.protocol.to_protocol_response(self.details, *response)
        else:
            response_converted = self.protocol.to_protocol_response(self.details, response)

        return response_converted

    def _create_remote(self) -> Remote:
        cls_path = self.model_spec.runtime_options.runtime
        if cls_path is None:
            if tempo_settings.use_kubernetes() or os.getenv(ENV_K8S_SERVICE_HOST):
                cls_path = self.model_spec.runtime_options.k8s_options.defaultRuntime
            else:
                cls_path = self.model_spec.runtime_options.docker_options.defaultRuntime
        logger.debug("Using remote class %s", cls_path)
        cls: Any = locate(cls_path)
        return cls()

    def remote(self, *args, **kwargs):
        remoter = self._create_remote()
        return remoter.remote(self.model_spec, *args, **kwargs)

    def wait_ready(self, runtime: Runtime, timeout_secs=None):
        return runtime.wait_ready_spec(self.model_spec, timeout_secs=timeout_secs)

    def get_endpoint(self, runtime: Runtime):
        return runtime.get_endpoint_spec(self.model_spec)

    def to_k8s_yaml(self, runtime: Runtime) -> str:
        """
        Get k8s yaml
        """

        return runtime.to_k8s_yaml_spec(self.model_spec)

    def deploy(self, runtime: Runtime):
        # self.set_runtime(runtime)
        runtime.deploy_spec(self.model_spec)

    def undeploy(self, runtime: Runtime):
        # self.unset_runtime()
        logger.info("Undeploying %s", self.details.name)
        runtime.undeploy_spec(self.model_spec)

    def get_tempo(self) -> BaseModel:
        return self

    def __call__(self, *args, **kwargs) -> Any:
        if not self._user_func:
            return self.remote(*args, **kwargs)
        else:
            if self.use_remote:
                return self.remote(*args, **kwargs)
            else:
                if self.cls is not None:
                    return self._user_func(self.cls, *args, **kwargs)
                else:
                    return self._user_func(*args, **kwargs)
