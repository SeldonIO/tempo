from __future__ import annotations

import abc
from typing import Any

import attr

from tempo.serve.metadata import ModelDetails
from tempo.serve.protocol import Protocol


@attr.s(auto_attribs=True)
class Runtime(abc.ABC):
    # TODO change to deploy_model
    @abc.abstractmethod
    def deploy(self, model_details: ModelDetails):
        pass

    # TODO change to undeploy_model
    @abc.abstractmethod
    def undeploy(self, model_details: ModelDetails):
        pass

    @abc.abstractmethod
    def remote(self, model_details: ModelDetails, *args, **kwargs) -> Any:
        pass

    @abc.abstractmethod
    def get_endpoint(self, model_details: ModelDetails) -> str:
        pass

    @abc.abstractmethod
    def wait_ready(self, model_details: ModelDetails, timeout_secs=None) -> bool:
        pass

    @abc.abstractmethod
    def get_protocol(self):
        pass

    # TODO change to to_yaml
    @abc.abstractmethod
    def to_k8s_yaml(self, model_details: ModelDetails) -> str:
        pass


class LocalRuntime(Runtime):
    """
    LocalRuntime lets you run model functions explicitly as a local function.
    """

    def __init__(self, protocol: Protocol):
        super().__init__()
        self._protocol = protocol

    def deploy(self, model_details: ModelDetails):
        pass

    def undeploy(self, model_details: ModelDetails):
        pass

    def remote(self, model_details: ModelDetails, *args, **kwargs) -> Any:
        raise NotImplementedError("LocalRuntime is only meant to be used locally")

    def get_endpoint(self, model_details: ModelDetails) -> str:
        return ""

    def wait_ready(self, model_details: ModelDetails, timeout_secs=None) -> bool:
        return True

    def get_protocol(self) -> Protocol:
        return self._protocol

    def to_k8s_yaml(self, model_details: ModelDetails) -> str:
        return ""
