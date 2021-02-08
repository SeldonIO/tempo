from typing import Callable, List, Any, Dict, Optional, Type, ClassVar
import types

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
            pipeline_cls: ClassVar = None,
            runtime: Runtime = None,
            models: List[Model] = None,
            inputs: ModelDataType = None,
            outputs: ModelDataType = None,
            remote_artifact_uri: str = None,
            local_artifact_folder: str = None,
    ):
        super().__init__(name,pipeline_func,runtime.get_protocol(),inputs,outputs)
        if models is None:
            models = []
        self._name = name
        self._pipeline_func = pipeline_func
        self._runtime = runtime
        self._models = models

    def deploy_models(self):
        """
        Deploy all the models
        """
        logger.info("deploying models for %s", self._name)
        for model in self._models:
            logger.info(f"Found model {model._details.name}")
            model.deploy()

    def deploy(self):
        """
        Deploy all models and the pipeline.
        """
        self.deploy_models()
        #TODO add deploy pipeline itself

    def undeploy_models(self):
        logger.info("undeploying models for %s", self._name)
        for model in self._models:
            model.undeploy()

    def undeploy(self):
        """
        Undeploy all models and pipeline.
        """
        self.undeploy_models()
        #TODO undeploy pipeline

    def __call__(self, raw: Any) -> Any:
        return self._pipeline_func(raw)

