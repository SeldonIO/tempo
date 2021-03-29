from typing import Any, Callable, List

from tempo.errors import UndefinedCustomImplementation
from tempo.serve.base import BaseModel
from tempo.serve.remote import Remote
from tempo.serve.constants import ModelDataType
from tempo.serve.metadata import ModelFramework
from tempo.serve.runtime import Runtime
from tempo.serve.protocol import Protocol


class Pipeline(BaseModel):
    def __init__(
        self,
        name: str,
        pipeline_func: Callable[[Any], Any] = None,
        runtime: Runtime = None,
        protocol: Protocol = None,
        models: List[BaseModel] = None,
        local_folder: str = None,
        uri: str = None,
        inputs: ModelDataType = None,
        outputs: ModelDataType = None,
        conda_env: str = None,
        deployed: bool = False,
    ):
        super().__init__(
            name=name,
            # TODO: Should we unify names?
            user_func=pipeline_func,
            local_folder=local_folder,
            uri=uri,
            platform=ModelFramework.TempoPipeline,
            inputs=inputs,
            outputs=outputs,
            conda_env=conda_env,
            deployed=deployed,
            protocol=protocol,
        )

        if models is None:
            models = []

        self._models = models

    def save(self, save_env=True):
        for model in self._models:
            model.set_deployed(True)
        super().save(save_env=save_env)
        for model in self._models:
            model.set_deployed(False)

    def deploy_models(self, runtime: Runtime):
        for model in self._models:
            model.deploy(runtime)

    def deploy(self, runtime: Runtime):
        self.deploy_models(runtime)
        super().deploy(runtime)

    def wait_ready(self, runtime:Runtime, timeout_secs: int = None) -> bool:
        super().wait_ready(runtime, timeout_secs=timeout_secs)
        for model in self._models:
            if not model.wait_ready(runtime, timeout_secs=timeout_secs):
                return False
        return True

    def undeploy_models(self, runtime: Runtime):
        for model in self._models:
            model.undeploy(runtime)

    def undeploy(self, runtime: Runtime):
        """
        Undeploy all models and pipeline.
        """
        super().undeploy(runtime)
        self.undeploy_models(runtime)

    def set_remote(self, runtime: Remote):
        super().set_remote(runtime)
        for model in self._models:
            model.set_remote(runtime)

    def to_k8s_yaml(self, runtime: Runtime) -> str:
        yamls = super().to_k8s_yaml(runtime)
        yamls += "\n---\n"
        for model in self._models:
            y = model.to_k8s_yaml(runtime)
            yamls += y
            yamls += "\n---\n"
        return yamls

    def __call__(self, *args, **kwargs) -> Any:
        if not self._user_func:
            # TODO: Group generic errors
            raise UndefinedCustomImplementation(self.details.name)

        if self.deployed:
            return self.remote(*args, **kwargs)
        else:
            if self.cls is not None:
                return self._user_func(self.cls, *args, **kwargs)
            else:
                return self._user_func(*args, **kwargs)
