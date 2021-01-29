from typing import Callable, List, Any, Dict, Optional, Type

from tempo.serve.constants import ModelDataType
from tempo.serve.model import Model
from tempo.utils import logger
from tempo.serve.runtime import Runtime
from tempo.serve.base import BaseModel


class Pipeline(BaseModel):
    def __init__(
        self,
            name: str,
            pipeline_func: Callable[[Any], Any],
            runtime: Runtime = None,
            models: List[Model] = None,
            inputs: ModelDataType = None,
            outputs: ModelDataType = None
    ):
        super().__init__(name,pipeline_func,runtime,inputs,outputs)
        if models is None:
            models = []
        self._name = name
        self._pipeline_func = pipeline_func
        self._runtime = runtime
        self._models = models


    def deploy(self):
        """
        Deploy all models and the pipeline.
        """
        logger.info("deploying models for %s", self._name)
        for model in self._models:
            logger.info(f"Found model {model._details.name}")
            model.deploy()

    def undeploy(self):
        """
        Undeploy all models and pipeline.
        """
        logger.info("undeploying models for %s", self._name)
        for model in self._models:
            model.undeploy()

    def __call__(self, raw: Any) -> Any:
        return self._pipeline_func(raw)
