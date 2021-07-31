from pydoc import locate
from typing import Any

from tempo.serve.base import Runtime
from tempo.serve.deploy import RemoteModel
from tempo.serve.metadata import BaseProductOptionsType, BaseRuntimeOptionsType, DockerOptions, KubernetesRuntimeOptions


class AsyncRemoteModel(RemoteModel):
    def __init__(self, model: Any, runtime: Runtime):
        super(AsyncRemoteModel, self).__init__(model, runtime)

    async def predict(self, *args, **kwargs):
        return await self.model.remote_with_spec(self.model_spec, *args, **kwargs)


def _get_runtime(cls_path, options: BaseRuntimeOptionsType) -> Runtime:
    cls: Any = locate(cls_path)
    return cls(options)


def deploy(model: Any, options: BaseRuntimeOptionsType = None) -> RemoteModel:
    if options is None:
        options = DockerOptions()
    rt: Runtime = _get_runtime(options.runtime, options)
    rm = RemoteModel(model, rt)
    rm.deploy()
    return rm


def deploy_local(model: Any, options: BaseProductOptionsType = None) -> RemoteModel:
    if options is None:
        runtime_options = DockerOptions()
    else:
        runtime_options = options.local_options
    rt: Runtime = _get_runtime(runtime_options.runtime, runtime_options)
    rm = RemoteModel(model, rt)
    rm.deploy()
    return rm


def deploy_remote(model: Any, options: BaseProductOptionsType = None) -> RemoteModel:
    if options is None:
        runtime_options = KubernetesRuntimeOptions()
    else:
        runtime_options = options.remote_options  # type: ignore
    rt: Runtime = _get_runtime(runtime_options.runtime, runtime_options)
    rm = RemoteModel(model, rt)
    rm.deploy()
    return rm
