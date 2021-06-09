from ..serve.model import Model as _Model
from .mixin import _AsyncMixin


class Model(_AsyncMixin, _Model):  # type: ignore
    def __init__(self, *args, **kwargs):
        _Model.__init__(self, *args, **kwargs)
        _AsyncMixin.__init__(self)
