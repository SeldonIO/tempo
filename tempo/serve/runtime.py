from __future__ import annotations

import abc
from pydoc import locate
from typing import Any, Optional, Dict

from pydantic import BaseModel, validator

from tempo.serve.metadata import ModelDetails, RuntimeOptions
from tempo.serve.protocol import Protocol
from tempo.serve.typing import fullname


class ModelSpec(BaseModel):

    model_details: ModelDetails
    protocol: Protocol
    runtime_options: RuntimeOptions

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
        return t.get_endpoint(self)

    def wait_ready(self, model: Any, timeout_secs=None):
        t = model.get_tempo()
        t.set_runtime_options_override(self.runtime_options)
        t.wait_ready(self, timeout_secs)

    def to_k8s_yaml(self, model: Any):
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
