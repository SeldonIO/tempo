from ..serve.pipeline import Pipeline as _Pipeline
from .mixin import _AsyncMixin


class Pipeline(_AsyncMixin, _Pipeline):
    def __init__(self, *args, **kwargs):
        _Pipeline.__init__(self, *args, **kwargs)
        _AsyncMixin.__init__(self)
