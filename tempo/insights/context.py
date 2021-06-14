import contextvars
from typing import Any

insights_context: Any = contextvars.ContextVar("insights_manager", default=None)


class classproperty(object):
    """
    Class function decorator for static class methods to behave
    like properties, but limited to only getter
    """

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class insights:
    @staticmethod
    def log(data):
        insights.context.log(data)  # pylint: disable=no-member

    @staticmethod
    def log_request():
        insights.context.log_request()  # pylint: disable=no-member

    @staticmethod
    def log_response():
        insights.context.log_response()

    @classproperty
    def context(cls):  # pylint: disable=no-self-argument
        return insights_context.get()
