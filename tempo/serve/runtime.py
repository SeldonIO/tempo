from __future__ import annotations
import abc
import attr
from tempo.serve.metadata import ModelDetails


@attr.s(auto_attribs=True)
class Runtime(abc.ABC):
    @abc.abstractmethod
    def deploy(self, model_details: ModelDetails):
        pass

    @abc.abstractmethod
    def undeploy(self, model_details: ModelDetails):
        pass

    @abc.abstractmethod
    def get_endpoint(self, model_details: ModelDetails):
        pass

    @abc.abstractmethod
    def get_protocol(self):
        pass


# @abc.abstractmethod
# def to_yaml(self) -> str:
#     pass
