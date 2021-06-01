from typing import Any
from inspect import iscoroutinefunction

from .model import Model
from ..errors import InvalidUserFunction


class AsyncModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        is_async = iscoroutinefunction(self._user_func)
        if not is_async:
            raise InvalidUserFunction(self._name, reason="function is not awaitable")

    async def __call__(self, *args, **kwargs) -> Any:
        future = super().__call__(*args, **kwargs)
        return await future
