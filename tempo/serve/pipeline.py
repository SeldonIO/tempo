from typing import Callable, List, Any

from tempo.serve.metadata import ModelFramework
from tempo.serve.constants import ModelDataType
from tempo.serve.model import Model
from tempo.utils import logger
from tempo.serve.runtime import Runtime
from tempo.serve.base import BaseModel


class Pipeline(BaseModel):
    def __init__(
        self,
        name: str,
        pipeline_func: Callable[[Any], Any] = None,
        runtime: Runtime = None,
        models: List[Model] = None,
        local_folder: str = None,
        uri: str = None,
        inputs: ModelDataType = None,
        outputs: ModelDataType = None,
        conda_env: str = None,
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
        )

        if models is None:
            models = []

        self._models = models

    def deploy_models(self):
        """
        Deploy all the models
        """
        logger.info("deploying models for %s", self.details.name)
        for model in self._models:
            logger.info(f"Found model {model.details.name}")
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
        logger.info("undeploying models for %s", self.details.name)
        for model in self._models:
            model.undeploy()

    def undeploy(self):
        """
        Undeploy all models and pipeline.
        """
        super().undeploy()
        self.undeploy_models()

    def set_runtime(self, runtime: Runtime):
        super().set_runtime(runtime)
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

    def __call__(self, raw: Any) -> Any:
        return self._user_func(raw)
