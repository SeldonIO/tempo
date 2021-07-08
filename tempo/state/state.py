import abc
from typing import Any, Dict, Optional

import attr

# TODO: Also support aioredis for when running inside mlserver
import redis

from tempo.serve.metadata import StateOptions, StateTypes


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

    @property
    @abc.abstractmethod
    def internal_state(self) -> Optional[Any]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def state_options(self) -> StateOptions:
        raise NotImplementedError

    @staticmethod
    def from_conf(state_options: StateOptions):
        state_type = state_options.state_type
        if state_type in (StateTypes.LOCAL, StateTypes.LOCAL.value):
            return LocalState(state_options=state_options)
        elif state_type in (StateTypes.REDIS, StateTypes.REDIS.value):
            return RedisState(state_options=state_options)
        else:
            raise Exception("State type not valid")


class LocalState(BaseState):
    def __init__(self, state_options: StateOptions = StateOptions()):
        self._internal_state: Dict = {}
        self._state_options = state_options

    def set(self, key: str, value: str) -> Optional[bool]:
        prefix = self._state_options.key_prefix
        self._internal_state[prefix + key] = value
        return True

    def get(self, key: str) -> Optional[Any]:
        prefix = self._state_options.key_prefix
        return self._internal_state.get(prefix + key)

    def exists(self, key: str) -> int:
        prefix = self._state_options.key_prefix
        return prefix + key in self._internal_state

    @property
    def internal_state(self) -> Dict:
        return self._internal_state

    @property
    def state_options(self):
        return self._state_options


class RedisState(BaseState):
    def __init__(self, state_options: StateOptions):
        self._state_options = state_options
        self._internal_state: redis.Redis = None  # type: ignore

    def _setup_state(self) -> None:

        self._redis_host = self._state_options.host
        self._redis_port = int(self._state_options.port)
        self._internal_state = redis.Redis(host=self._redis_host, port=self._redis_port)

    def set(self, key: str, value: str) -> Optional[bool]:
        prefix = self._state_options.key_prefix
        return self.internal_state.set(prefix + key, value)

    def get(self, key: str) -> Optional[Any]:
        prefix = self._state_options.key_prefix
        return self.internal_state.get(prefix + key)

    def exists(self, key: str) -> int:
        prefix = self._state_options.key_prefix
        return self.internal_state.exists(prefix + key)

    @property
    def internal_state(self) -> redis.Redis:
        if not self._internal_state:
            self._setup_state()
        return self._internal_state

    @property
    def state_options(self):
        return self._state_options
