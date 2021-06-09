from ..serve.pipeline import Pipeline as _Pipeline
from ..serve.pipeline import PipelineModels as _PipelineModels
from .mixin import _AsyncMixin
from .model import Model


class PipelineModels(_PipelineModels):
    # Use AsyncIO Model to export PipelineModels
    ModelExportKlass = Model


class Pipeline(_AsyncMixin, _Pipeline):  # type: ignore
    def __init__(self, *args, **kwargs):
        _Pipeline.__init__(self, *args, **kwargs)
        _AsyncMixin.__init__(self)

        # TODO: Do we need to convert models to async models on-the-fly?
        self.models = PipelineModels(**self.models.__dict__)
