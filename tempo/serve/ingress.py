import abc
from pydoc import locate
from typing import Any

import attr

from tempo.serve.runtime import ModelSpec


@attr.s(auto_attribs=True)
class Ingress(abc.ABC):
    @abc.abstractmethod
    def get_external_host_url(self, model_spec: ModelSpec) -> str:
        pass


def create_ingress(model_spec: ModelSpec) -> Ingress:
    cls: Any = locate(model_spec.runtime_options.ingress_options.ingress)
    return cls()
