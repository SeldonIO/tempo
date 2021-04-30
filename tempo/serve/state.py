import abc
import os
from typing import Any, Dict, Optional

import attr
import redis

from tempo.serve.metadata import ModelDetails, StateDetails, StateType


@attr.s(auto_attribs=True)
class BaseState(abc.ABC):
    @abc.abstractmethod
    def set(self, key: str, value: str) -> Optional[bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, key: str) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_internal_state(self) -> Optional[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_state_details(self) -> StateDetails:
        raise NotImplementedError


class LocalState(BaseState):
    def __init__(self, state_details: StateDetails = None, model_details: ModelDetails = None):
        self._internal_state: Dict = {}

    def set(self, key: str, value: str) -> Optional[bool]:
        self._internal_state[key] = value
        return True

    def get(self, key: str) -> Optional[Any]:
        return self._internal_state.get(key)

    def exists(self, key: str) -> int:
        return key in self._internal_state

    def get_internal_state(self) -> Dict:
        return self._internal_state

    def get_state_details(self):
        return StateDetails(
            state_type=StateType.local,
            config={},
        )


class DistributedState(BaseState):

    REDIS_HOST_ENV_NAME = "TEMPO_STATE_REDIS_HOST"
    REDIS_PORT_ENV_NAME = "TEMPO_STATE_REDIS_PORT"

    def __init__(self, state_details: StateDetails, model_details: ModelDetails = None):
        self._validate_required_params(state_details, model_details)
        self._state_details = state_details
        self._model_details = model_details
        self._internal_state: redis.Redis = None  # type: ignore

    def _setup_state(
        self, state_details_override: StateDetails = None, model_details_override: ModelDetails = None
    ) -> None:

        if state_details_override:
            self._state_details = state_details_override
        if model_details_override:
            self._model_details = model_details_override

        host = self._state_details.config["host"]
        port = self._state_details.config["port"]

        self._redis_host = os.environ.get(DistributedState.REDIS_HOST_ENV_NAME, host)
        self._redis_port = int(os.environ.get(DistributedState.REDIS_PORT_ENV_NAME, port))
        self._internal_state = redis.Redis(host=self._redis_host, port=self._redis_port)

    def set(self, key: str, value: str) -> Optional[bool]:
        return self.get_internal_state().set(key, value)

    def get(self, key: str) -> Optional[Any]:
        return self.get_internal_state().get(key)

    def exists(self, key: str) -> int:
        return self.get_internal_state().exists(key)

    def _validate_required_params(self, state_details: StateDetails, model_details: ModelDetails = None):
        # TODO: Validate state and model details for required fields
        pass

    def get_internal_state(self) -> redis.Redis:
        if not self._internal_state:
            self._setup_state()
        return self._internal_state

    def get_state_details(self):
        return StateDetails(
            state_type=StateType.redis,
            key_override=self._key_prefix,
            config=self._state_details.config,
        )
