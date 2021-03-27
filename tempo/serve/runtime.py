from __future__ import annotations

import abc
from typing import Any
from pydantic import BaseModel
import attr

from tempo.serve.metadata import ModelDetails
from tempo.serve.protocol import Protocol


class ModelSpec(BaseModel):

    model_details: ModelDetails
    protocol: Protocol

    class Config:
        arbitrary_types_allowed = True

@attr.s(auto_attribs=True)
class Runtime(abc.ABC):
    # TODO change to deploy_model
    @abc.abstractmethod
    def deploy(self, model_spec: ModelSpec):
        pass

    # TODO change to undeploy_model
    @abc.abstractmethod
    def undeploy(self, model_spec: ModelSpec):
        pass

    @abc.abstractmethod
    def remote(self, model_spec: ModelSpec, *args, **kwargs) -> Any:
        pass

    @abc.abstractmethod
    def get_endpoint(self, model_spec: ModelSpec) -> str:
        pass

    @abc.abstractmethod
    def wait_ready(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        pass

    # TODO change to to_yaml
    @abc.abstractmethod
    def to_k8s_yaml(self, model_spec: ModelSpec) -> str:
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
