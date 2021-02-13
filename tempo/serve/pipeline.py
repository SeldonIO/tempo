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
    ):
        super().__init__(
            name=name,
            # TODO: Should we unify names?
            user_func=pipeline_func,
            # TODO: What if `runtime` is None?
            protocol=runtime.get_protocol(),
            local_folder=local_folder,
            uri=uri,
            platform=ModelFramework.TempoPipeline,
            inputs=inputs,
            outputs=outputs,
        )

        if models is None:
            models = []

        self._runtime = runtime
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
        # TODO add deploy pipeline itself

    def undeploy_models(self):
        logger.info("undeploying models for %s", self.details.name)
        for model in self._models:
            model.undeploy()

    def undeploy(self):
        """
        Undeploy all models and pipeline.
        """
        self.undeploy_models()
        # TODO undeploy pipeline

    def __call__(self, raw: Any) -> Any:
        return self._user_func(raw)
