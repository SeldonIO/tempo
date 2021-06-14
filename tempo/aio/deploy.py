from pydoc import locate
from typing import Any

from tempo.serve.base import Runtime
from tempo.serve.deploy import RemoteModel
from tempo.serve.metadata import RuntimeOptions


class AsyncRemoteModel(RemoteModel):
    def __init__(self, model: Any, runtime: Runtime):
        super(AsyncRemoteModel, self).__init__(model, runtime)

    async def predict(self, *args, **kwargs):
        return await self.model.remote_with_spec(self.model_spec, *args, **kwargs)


def _get_runtime(cls_path, options: RuntimeOptions) -> Runtime:
    cls: Any = locate(cls_path)
    return cls(options)


def deploy(model: Any, options: RuntimeOptions = None) -> RemoteModel:
    if options is None:
        options = RuntimeOptions()
    rt: Runtime = _get_runtime(options.runtime, options)
    rm = RemoteModel(model, rt)
    rm.deploy()
    return rm
