from __future__ import annotations

import abc
from typing import Any, Optional

from pydantic import BaseModel

from tempo.serve.metadata import ModelDetails, RuntimeOptions
from tempo.serve.protocol import Protocol


class ModelSpec(BaseModel):

    model_details: ModelDetails
    protocol: Protocol
    runtime_options: RuntimeOptions

    class Config:
        arbitrary_types_allowed = True


class Deployer(object):
    def __init__(self, runtime_options: Optional[RuntimeOptions]):
        self.runtime_options = runtime_options

    def deploy(self, model: Any):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        t.deploy(self)

    def undeploy(self, model: Any):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        t.undeploy(self)

    def get_endpoint(self, model: Any):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        t.get_endpoint(self)

    def wait_ready(self, model: Any, timeout_secs=None):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        t.wait_ready(self, timeout_secs)

    def to_k8s_yaml(self, model: Any):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        return t.to_k8s_yaml(self)


class Runtime(abc.ABC, Deployer):
    # TODO change to deploy_model
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

    @abc.abstractmethod
    def wait_ready_spec(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        pass

    # TODO change to to_yaml
    @abc.abstractmethod
    def to_k8s_yaml_spec(self, model_spec: ModelSpec) -> str:
        pass


class LocalRuntime(Runtime):
    """
    LocalRuntime lets you run model functions explicitly as a local function.
    """

    def __init__(self):
        super().__init__()

    def deploy(self, model_details: ModelDetails):
        pass

    def undeploy(self, model_spec: ModelDetails):
        pass

    def remote(self, model_spec: ModelDetails, *args, **kwargs) -> Any:
        raise NotImplementedError("LocalRuntime is only meant to be used locally")

    def get_endpoint(self, model_spec: ModelDetails) -> str:
        return ""

    def wait_ready(self, model_spec: ModelDetails, timeout_secs=None) -> bool:
        return True

    def to_k8s_yaml(self, model_spec: ModelDetails) -> str:
        return ""
