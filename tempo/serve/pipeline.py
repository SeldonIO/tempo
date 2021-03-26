from typing import Any, Callable, List

from tempo.errors import UndefinedCustomImplementation
from tempo.serve.base import BaseModel
from tempo.serve.constants import ModelDataType
from tempo.serve.metadata import ModelFramework
from tempo.serve.runtime import Runtime
from tempo.utils import logger


class Pipeline(BaseModel):
    def __init__(
        self,
        name: str,
        pipeline_func: Callable[[Any], Any] = None,
        runtime: Runtime = None,
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
            runtime=runtime,
            deployed=deployed,
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

    def deploy_models(self):
        """
        Deploy all the models
        """
        for model in self._models:
            model.deploy()

    def deploy(self):
        """
        Deploy all models and the pipeline.
        """
        self.deploy_models()
        super().deploy()

    def wait_ready(self, timeout_secs: int = None) -> bool:
        super().wait_ready(timeout_secs=timeout_secs)
        for model in self._models:
            if not model.wait_ready(timeout_secs=timeout_secs):
                return False
        return True

    def undeploy_models(self):
        for model in self._models:
            model.undeploy()

    def undeploy(self):
        """
        Undeploy all models and pipeline.
        """
        super().undeploy()
        self.undeploy_models()

    def set_runtime(self, runtime: Runtime, models=False):
        super().set_runtime(runtime)
        if models:
            for model in self._models:
                model.set_runtime(runtime)

    def to_k8s_yaml(self) -> str:
        yamls = super().to_k8s_yaml()
        yamls += "\n---\n"
        for model in self._models:
            y = model.to_k8s_yaml()
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
