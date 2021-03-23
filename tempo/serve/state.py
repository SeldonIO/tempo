
import abc
import attr
import os
import redis

from pydantic import BaseModel
from enum import Enum
from typing import List, Dict

from tempo.serve.metadata import ModelDetails, StateType, StateDetails

@attr.s(auto_attribs=True)
class BaseState(abc.ABC):

    @abc.abstractmethod
    def set(self, key: str, value: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, key: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_internal_state(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_state_details(self) -> StateDetails:
        raise NotImplementedError

class LocalState(BaseState):

    def __init__(self, state_details: StateDetails = None, model_details: ModelDetails = None):
        self._internal_state = {}

    def set(self, key: str, value: str) -> bool:
        self._internal_state[key] = value
        return True

    def get(self, key: str) -> str:
        return self._internal_state.get(key)

    def exists(self, key: str) -> bool:
        return key in self._internal_state

    def get_internal_state(self):
        return self._internal_state

    def get_state_details(self):
        return StateDetails(
            state_type=StateType.local,
            key_override="",
            config={},
        )

class DistributedState(BaseState):

    REDIS_HOST_ENV_NAME = "TEMPO_STATE_REDIS_HOST"
    REDIS_PORT_ENV_NAME = "TEMPO_STATE_REDIS_PORT"
    REDIS_KEY_ENV_NAME = "TEMPO_STATE_REDIS_KEY"

    def __init__(self, state_details: StateDetails, model_details: ModelDetails = None):
        self._validate_required_params(state_details, model_details)
        self._state_details = state_details
        self._model_details = model_details
        self._key_prefix = state_details.key_override or model_details.uri
        self._internal_state = None

    def _setup_state(self,
                     state_details_override: StateDetails = None,
                     model_details_override: ModelDetails = None) -> None:

        if state_details_override: self._state_details = state_details_override
        if model_details_override: self._model_details = model_details_override

        host = self._state_details.config["host"]
        port = self._state_details.config["port"]
        key_prefix = self._state_details.key_override or self._model_details.uri

        self._redis_host = os.environ.get(DistributedState.REDIS_HOST_ENV_NAME, host)
        self._redis_port = int(os.environ.get(DistributedState.REDIS_PORT_ENV_NAME, port))
        self._internal_state = redis.Redis(host=self._redis_host, port=self._redis_port)
        self._key_prefix = os.environ.get(DistributedState.REDIS_KEY_ENV_NAME, key_prefix)

    def set(self, key: str, value: str) -> bool:
        return self.get_internal_state().set(self._key_prefix + key, value)

    def get(self, key: str) -> str:
        return self.get_internal_state().get(self._key_prefix + key)

    def exists(self, key: str) -> bool:
        return self.get_internal_state().exists(self._key_prefix + key)

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

