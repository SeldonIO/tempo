from __future__ import annotations

import abc
import os
import tempfile
import uuid
from pydoc import locate
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import numpy as np
import pydantic
import requests
from pydantic import validator

from ..errors import UndefinedCustomImplementation
from ..insights.manager import InsightsManager
from ..magic import PayloadContext, TempoContextWrapper, tempo_context
from ..state.state import BaseState
from ..utils import logger
from .args import infer_args, process_datatypes
from .constants import DefaultCondaFile, DefaultEnvFilename, DefaultModelFilename
from .loader import load_custom, save_custom, save_environment
from .metadata import (
    BaseRuntimeOptionsType,
    ClientDetails,
    DockerOptions,
    InsightRequestModes,
    ModelDataArg,
    ModelDataArgs,
    ModelDetails,
    ModelFramework,
)
from .protocol import Protocol
from .types import LoadMethodSignature, ModelDataType, PredictMethodSignature
from .typing import fullname


class BaseModel:
    def __init__(
        self,
        name: str,
        user_func: PredictMethodSignature = None,
        load_func: LoadMethodSignature = None,
        local_folder: str = None,
        uri: str = None,
        platform: ModelFramework = None,
        inputs: ModelDataType = None,
        outputs: ModelDataType = None,
        conda_env: str = None,
        protocol: Protocol = None,
        runtime_options: BaseRuntimeOptionsType = DockerOptions(),
        model_spec: ModelSpec = None,
        description: str = "",
    ):
        if model_spec is not None:
            self.name = model_spec.model_details.name
            self.details = model_spec.model_details
            self.model_spec = model_spec
        else:
            self._name = name
            self._user_func = user_func
            self._load_func = load_func
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
                description=description,
            )

            self.model_spec = ModelSpec(
                model_details=self.details,
                protocol=protocol,
                runtime_options=runtime_options,
            )

        self.use_remote: bool = False
        self.runtime_options_override: Optional[BaseRuntimeOptionsType] = None

        insights_params = runtime_options.insights_options.dict()
        self.insights_manager = InsightsManager(**insights_params)

        self.state = BaseState.from_conf(runtime_options.state_options)

        # K holds the wrapped class (if any)
        self._K: Optional[Type] = None

        # context represents internal context shared (optionally) between different
        # methods of the model (e.g. predict, loader, etc.)
        self.context = SimpleNamespace()

    def set_remote(self, val: bool):
        self.use_remote = val

    def set_runtime_options_override(self, runtime_options: BaseRuntimeOptionsType):
        self.runtime_options_override = runtime_options

    def _get_args(
        self, inputs: ModelDataType = None, outputs: ModelDataType = None
    ) -> Tuple[ModelDataArgs, ModelDataArgs]:
        if isinstance(inputs, ModelDataArgs) and isinstance(outputs, ModelDataArgs):
            return inputs, outputs

        if inputs or outputs:
            return process_datatypes(inputs, outputs)

        if self._user_func is not None:
            return infer_args(self._user_func)

        default_input_args = ModelDataArgs(args=[ModelDataArg(ty=np.ndarray)])
        default_output_args = ModelDataArgs(args=[ModelDataArg(ty=np.ndarray)])
        return default_input_args, default_output_args

    def _get_local_folder(self, local_folder: str = None) -> Optional[str]:
        if not local_folder:
            # TODO: Should we do cleanup at some point?
            local_folder = tempfile.mkdtemp()

        return local_folder

    def loadmethod(self, load_func: LoadMethodSignature) -> LoadMethodSignature:
        self._load_func = load_func
        self._load_func()

        return load_func

    def __getstate__(self) -> dict:
        """
        __getstate__ gets called by pickle before serialising an object to get
        its internal representation.
        We override __getstate__ to make sure that the model's internal context
        is not pickled with the object.
        """
        state = self.__dict__.copy()
        state["context"] = SimpleNamespace()
        # Remove the insights manager from the cloudpickle context
        state["insights_manager"] = SimpleNamespace()
        state["state"] = SimpleNamespace()

        return state

    @classmethod
    def load(cls, folder: str) -> "BaseModel":
        file_path_pkl = os.path.join(folder, DefaultModelFilename)
        if os.path.exists(file_path_pkl):
            return load_custom(file_path_pkl)
        else:
            raise ValueError("This is not a custom tempo model and cannot be loaded with this mechanism")

    def save(self, save_env=True) -> None:
        logger.info("Saving environment")
        # we want to pack the conda environment anyway
        if save_env:
            file_path_env = os.path.join(self.details.local_folder, DefaultEnvFilename)
            conda_env_file_path = os.path.join(self.details.local_folder, DefaultCondaFile)

            save_environment(
                conda_pack_file_path=file_path_env,
                conda_env_file_path=conda_env_file_path if os.path.exists(conda_env_file_path) else None,
                env_name=self.conda_env_name,
                platform=self.details.platform,
            )

        if self._user_func:
            file_path_pkl = os.path.join(self.details.local_folder, DefaultModelFilename)
            logger.info("Saving tempo model to %s", file_path_pkl)
            if self._user_func is not None:
                module = self._user_func.__module__
            else:
                module = None
            save_custom(self, module, file_path_pkl)

    def request(self, req: Dict) -> Dict:

        if self._user_func is None:
            raise UndefinedCustomImplementation(self.details.name)

        req_converted = self.model_spec.protocol.from_protocol_request(req, self.details.inputs)
        if type(req_converted) == dict:
            response = self(**req_converted)
        elif type(req_converted) == list or type(req_converted) == tuple:
            response = self(*req_converted)
        else:
            response = self(req_converted)

        if type(response) == dict:
            response_converted = self.model_spec.protocol.to_protocol_response(self.details, **response)
        elif type(response) == list or type(response) == tuple:
            response_converted = self.model_spec.protocol.to_protocol_response(self.details, *response)
        else:
            response_converted = self.model_spec.protocol.to_protocol_response(self.details, response)

        return response_converted

    def _get_model_spec(self, runtime: Optional[Runtime]) -> ModelSpec:
        if self.runtime_options_override:
            return ModelSpec(
                model_details=self.model_spec.model_details,
                protocol=self.model_spec.protocol,
                runtime_options=self.runtime_options_override,
            )
        elif runtime is not None:
            return ModelSpec(
                model_details=self.model_spec.model_details,
                protocol=self.model_spec.protocol,
                runtime_options=runtime.runtime_options,
            )
        else:
            return self.model_spec

    def _create_remote(self, model_spec: ModelSpec) -> Runtime:
        cls_path = model_spec.runtime_options.runtime
        logger.debug("Using remote class %s", cls_path)
        cls: Any = locate(cls_path)
        return cls(model_spec.runtime_options)

    def get_insights_mode(self) -> InsightRequestModes:
        return self.model_spec.runtime_options.insights_options.mode_type

    def predict(self, *args, **kwargs):
        # TODO: Decouple to support multiple transports (e.g. Kafka, gRPC)
        model_spec = self._get_model_spec(None)
        return self.remote_with_spec(model_spec, *args, **kwargs)

    def remote_with_client(self, model_spec: ModelSpec, client_details: ClientDetails, *args, **kwargs):
        prot = model_spec.protocol
        req = prot.to_protocol_request(*args, **kwargs)
        logger.debug(
            "Calling requests POST with client details endpoint=%s headers=%s verify=%s",
            client_details.url,
            client_details.headers,
            client_details.verify_ssl,
        )
        response_raw = requests.post(
            client_details.url, json=req, headers=client_details.headers, verify=client_details.verify_ssl
        )
        logger.debug(response_raw.content)

        response_raw.raise_for_status()

        response_json = response_raw.json()
        logger.debug("Response raw %s", response_json)
        output_schema = model_spec.model_details.outputs

        return prot.from_protocol_response(response_json, output_schema)

    def remote_with_spec(self, model_spec: ModelSpec, *args, **kwargs):
        remoter = self._create_remote(model_spec)
        ingress_options = model_spec.runtime_options.ingress_options
        endpoint = remoter.get_endpoint_spec(model_spec)
        headers = remoter.get_headers(model_spec)

        prot = model_spec.protocol
        req = prot.to_protocol_request(*args, **kwargs)
        logger.debug(
            "Calling requests POST with endpoint=%s headers=%s verify=%s", endpoint, headers, ingress_options.verify_ssl
        )
        response_raw = requests.post(endpoint, json=req, headers=headers, verify=ingress_options.verify_ssl)
        logger.debug(response_raw.content)

        response_raw.raise_for_status()

        response_json = response_raw.json()
        output_schema = model_spec.model_details.outputs

        res = prot.from_protocol_response(response_json, output_schema)
        logger.debug("protocol decoded %s", res)
        return res

    def wait_ready(self, runtime: Runtime, timeout_secs=None):
        return runtime.wait_ready_spec(self._get_model_spec(runtime), timeout_secs=timeout_secs)

    def get_endpoint(self, runtime: Runtime):
        return runtime.get_endpoint_spec(self._get_model_spec(runtime))

    def to_k8s_yaml(self, runtime: Runtime) -> str:
        """
        Get k8s yaml
        """

        return runtime.to_k8s_yaml_spec(self._get_model_spec(runtime))

    def deploy(self, runtime: Runtime):
        # self.set_runtime(runtime)
        runtime.deploy_spec(self._get_model_spec(runtime))

    def undeploy(self, runtime: Runtime):
        # self.unset_runtime()
        logger.info("Undeploying %s", self.details.name)
        runtime.undeploy_spec(self._get_model_spec(runtime))

    def serialize(self) -> str:
        return self._get_model_spec(None).json()

    def get_tempo(self) -> BaseModel:
        return self

    def __call__(self, *args, **kwargs) -> Any:
        if self._user_func is None:
            return self.predict(*args, **kwargs)

        if self.use_remote:
            return self.predict(*args, **kwargs)

        # When calling the method from outside mlserver the context is not set
        # In this situation the context has to be set manually to the local created
        if not tempo_context.get():
            logger.debug("Setting context to context for insights manager")
            # Initialising with unique ID as request not provided by server
            payload_context = PayloadContext(request_id=str(uuid.uuid4()))
            tempo_wrapper = TempoContextWrapper(payload_context, self.insights_manager, self.state)
            tempo_context.set(tempo_wrapper)

        return self._user_func(*args, **kwargs)


class ClientModel(BaseModel):
    def __init__(self, model_spec: ModelSpec, client_details: ClientDetails = None):
        super().__init__(model_spec.model_details.name, model_spec=model_spec)
        self.client_details = client_details

    def predict(self, *args, **kwargs):
        if self.client_details is not None:
            return super().remote_with_client(self.model_spec, self.client_details, *args, **kwargs)
        else:
            return super().predict(*args, **kwargs)

    def deploy(self, runtime: Runtime):
        logger.warn("Remote model %s can't be deployed", self.model_spec.model_details.name)
        pass

    def undeploy(self, runtime: Runtime):
        logger.warn("Remote model %s can't be undeployed", self.model_spec.model_details.name)
        pass


class ModelSpec(pydantic.BaseModel):

    model_details: ModelDetails
    protocol: Protocol
    runtime_options: BaseRuntimeOptionsType

    @validator("protocol", pre=True)
    def ensure_type(cls, v):
        if isinstance(v, str):
            klass = locate(v)
            return klass()
        else:
            return v

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Protocol: lambda v: fullname(v),
            type: lambda v: v.__module__ + "." + v.__name__,
        }


class Deployer(object):
    def __init__(self, runtime_options: Optional[BaseRuntimeOptionsType]):
        self.runtime_options = runtime_options

    def deploy(self, model: Any):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        t.deploy(self)

    def undeploy(self, model: Any):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        t.undeploy(self)

    def endpoint(self, model: Any):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        return t.get_endpoint(self)

    def wait_ready(self, model: Any, timeout_secs=None):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        t.wait_ready(self, timeout_secs)

    def manifest(self, model: Any):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        return t.to_k8s_yaml(self)


class Runtime(abc.ABC, Deployer):
    @abc.abstractmethod
    def deploy_spec(self, model_spec: ModelSpec):
        pass

    # TODO change to undeploy_model
    @abc.abstractmethod
    def undeploy_spec(self, model_spec: ModelSpec):
        pass

    @abc.abstractmethod
    def get_endpoint_spec(self, model_spec: ModelSpec) -> str:
        pass

    def get_headers(self, model_spec: ModelSpec) -> Dict[str, str]:
        return {}

    @abc.abstractmethod
    def wait_ready_spec(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        pass

    # TODO change to to_yaml
    @abc.abstractmethod
    def to_k8s_yaml_spec(self, model_spec: ModelSpec) -> str:
        pass

    @abc.abstractmethod
    def list_models(self) -> Sequence[ClientModel]:
        pass
