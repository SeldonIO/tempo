import abc
from typing import Any, Optional

from tempo.serve.metadata import RuntimeOptions
from tempo.serve.runtime import ModelSpec


class Remote(abc.ABC):
    def __init__(self, runtime_options: Optional[RuntimeOptions]):
        self.runtime_options = runtime_options

    @abc.abstractmethod
    def remote(self, model_spec: ModelSpec, *args, **kwargs) -> Any:
        pass
