from pydoc import locate
from typing import Any

from .base import BaseModel, ModelSpec, Runtime
from .metadata import RuntimeOptions


class RemoteModel:
    def __init__(self, model: Any, runtime: Runtime):
        self.model: BaseModel = model.get_tempo()
        self.runtime = runtime
        self.model_spec = ModelSpec(
            model_details=self.model.model_spec.model_details,
            protocol=self.model.model_spec.protocol,
            runtime_options=self.runtime.runtime_options,
        )

    def deploy(self):
        self.model.deploy(self.runtime)
        self.model.wait_ready(self.runtime)

    def predict(self, *args, **kwargs):
        return self.model.remote_with_spec(self.model_spec, *args, **kwargs)

    def endpoint(self):
        return self.model.get_endpoint(self.runtime)

    def manifest(self):
        return self.model.to_k8s_yaml(self.runtime)

    def undeploy(self):
        self.model.undeploy(self.runtime)


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
