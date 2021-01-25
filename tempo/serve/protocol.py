from __future__ import annotations

import abc
import attr

from typing import Dict, Type, Any, Optional, List


@attr.s(auto_attribs=True)
class Protocol(abc.ABC):

    @abc.abstractmethod
    def to_protocol_request(self, *args, **kwargs) -> Dict:
        pass

    @abc.abstractmethod
    def to_protocol_response(self, *args, **kwargs) -> Dict:
        pass

    @abc.abstractmethod
    def from_protocol_request(self, res: Dict, tys: List[Type]) -> Any:
        pass

    @abc.abstractmethod
    def from_protocol_response(self, res: Dict, tys: List[Type]) -> Any:
        pass

    @abc.abstractmethod
    def get_predict_path(self) -> str:
        pass
