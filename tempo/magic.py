import contextvars
from typing import Any, Optional

from pydantic import BaseModel

tempo_context: Any = contextvars.ContextVar("tempo_context", default=None)


class classproperty(object):
    """
    Class function decorator for static class methods to behave
    like properties, but limited to only getter
    """

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class tempo:
    @classproperty
    def insights(cls):  # pylint: disable=no-self-argument
        return tempo.context.insights  # pylint: disable=no-member

    @classproperty
    def payload(cls):  # pylint: disable=no-self-argument
        return tempo.context.payload  # pylint: disable=no-member

    @classproperty
    def context(cls):  # pylint: disable=no-self-argument
        return tempo_context.get()


class PayloadContext(BaseModel):
    request_id: Optional[str] = None
    request_headers: Optional[dict] = None
    request: Optional[dict] = None
    response_headers: Optional[dict] = None


class TempoContextWrapper:
    def __init__(self, payload_context, insights_worker):
        self.payload = payload_context
        self.insights = insights_worker
