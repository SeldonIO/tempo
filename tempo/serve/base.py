import os
import tempfile
from os import path
from typing import Any, Callable, Dict, Optional, Tuple, get_type_hints

from tempo.errors import UndefinedCustomImplementation
from tempo.serve.constants import DefaultEnvFilename, DefaultModelFilename, ModelDataType, \
    DefaultCondaFile
from tempo.serve.loader import download, load_custom, save_custom, save_environment, upload
from tempo.serve.metadata import ModelDataArg, ModelDataArgs, ModelDetails, ModelFramework
from tempo.serve.remote import Remote
from tempo.serve.runtime import Runtime, ModelSpec
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.utils import logger
from tempo.serve.protocol import Protocol


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
        protocol: Protocol = KFServingV2Protocol(),
        deployed: bool = False,
    ):
        self._name = name
        self._user_func = user_func
        self.conda_env_name = conda_env
        self.deployed = deployed

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
        self.model_spec = ModelSpec(
            model_details = self.details,
            protocol = self.protocol
        )
        self.remoter = None

    def set_deployed(self, val: bool):
        self.deployed = val

    def _get_args(
        self, inputs: ModelDataType = None, outputs: ModelDataType = None
    ) -> Tuple[ModelDataArgs, ModelDataArgs]:
        input_args = []
        output_args = []

        if inputs is None and outputs is None:
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
        logger.info("Saving tempo model to %s",file_path_pkl)
        if not self.deployed:
            self.deployed = True
            save_custom(self, file_path_pkl)
            self.deployed = False
        else:
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

    #def upload(self):
    #    """
    #    Upload from local folder to uri
    #    """
    #    upload(self.details.local_folder, self.details.uri)

    #def download(self):
    #    """
    #    Download from uri to local folder
    #    """
    #    # TODO: This doesn't make sense for custom methods?
    #    download(self.details.uri, self.details.local_folder)

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

    def set_remote(self, remote: Remote):
        self.remoter = remote

    def remote(self, *args, **kwargs):
        return self.remoter.remote(self.model_spec, *args, **kwargs)

    def wait_ready(self, runtime:Runtime, timeout_secs=None):
        return runtime.wait_ready_spec(self.model_spec, timeout_secs=timeout_secs)

    def get_endpoint(self, runtime: Runtime):
        return runtime.get_endpoint_spec(self.model_spec)

    def to_k8s_yaml(self, runtime: Runtime) -> str:
        """
        Get k8s yaml
        """

        return runtime.to_k8s_yaml_spec(self.model_spec)

    def deploy(self, runtime: Runtime):
        runtime.deploy_spec(self.model_spec)

    def undeploy(self, runtime: Runtime):
        logger.info("Undeploying %s", self.details.name)
        runtime.undeploy_spec(self.model_spec)


    def get_tempo(self):
        return self
