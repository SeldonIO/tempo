from __future__ import annotations

import abc
from typing import Any, Dict

import attr

from tempo.serve.metadata import ModelDataArgs, ModelDetails


@attr.s(auto_attribs=True)
class Protocol(abc.ABC):
    @abc.abstractmethod
    def to_protocol_request(self, *args, **kwargs) -> Dict:
        pass

    @abc.abstractmethod
    def to_protocol_response(self, model_details: ModelDetails, *args, **kwargs) -> Dict:
        pass

    @abc.abstractmethod
    def from_protocol_request(self, res: Dict, tys: ModelDataArgs) -> Any:
        pass

    @abc.abstractmethod
    def from_protocol_response(self, res: Dict, tys: ModelDataArgs) -> Any:
        pass

    @abc.abstractmethod
    def get_predict_path(self, model_details: ModelDetails) -> str:
        pass

    @abc.abstractmethod
    def get_status_path(self, model_details: ModelDetails) -> str:
        pass
