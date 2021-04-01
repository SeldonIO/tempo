import abc
from typing import Any

import attr

from tempo.serve.runtime import ModelSpec


@attr.s(auto_attribs=True)
class Remote(abc.ABC):
    def set_remote(self, model: Any):
        t = model.get_tempo()
        t.set_remote(self)

    @abc.abstractmethod
    def remote(self, model_spec: ModelSpec, *args, **kwargs) -> Any:
        pass
