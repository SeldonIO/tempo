from typing import Callable, List, Any, Dict, Optional, Type

from tempo.serve.model import Model
from tempo.serve.loader import load_pipeline, save_pipeline
from tempo.utils import logger
from tempo.serve.runtime import Runtime
from tempo.serve.metadata import ModelFramework, MetadataTensor


# TODO: Should Pipeline inherit from Model?
class Pipeline:
    def __init__(
        self,
            name: str,
            pipeline_func: Callable[[Any], Any],
            runtime: Runtime = None,
            models: List[Model] = None,
            inputs: List[MetadataTensor] = None,
            outputs: List[MetadataTensor] = None
    ):
        if models is None:
            models = []
        self._name = name
        self._pipeline_func = pipeline_func
        self._runtime = runtime
        self._models = models
        self._inputs = inputs
        self._outputs = outputs

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

    @classmethod
    def load(cls, file_path: str) -> "Pipeline":
        return load_pipeline(file_path)

    def save(self, file_path: str):
        save_pipeline(self, file_path)

    def request(self, req: Dict) -> Dict:
        protocol = self._runtime.get_protocol()
        tys = self._get_input_types()
        req_converted = protocol.from_protocol_request(req, tys)
        response = self._pipeline_func(req_converted)
        response_converted = protocol.to_protocol_response(response)
        return response_converted

    def _get_input_types(self) -> List[Type]:
        if not self._inputs:
            return []

        tys = []
        for output_meta in self._inputs:
            parameters = output_meta.parameters
            if not parameters:
                tys.append(None)
            else:
                tys.append(parameters.ext_datatype)
        return tys

    def __call__(self, raw: Any) -> Any:
        return self._pipeline_func(raw)
