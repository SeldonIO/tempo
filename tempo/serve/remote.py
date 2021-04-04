import abc
from typing import Any

import attr

from tempo.serve.runtime import ModelSpec


@attr.s(auto_attribs=True)
class Remote(abc.ABC):
    @abc.abstractmethod
    def remote(self, model_spec: ModelSpec, *args, **kwargs) -> Any:
        pass
