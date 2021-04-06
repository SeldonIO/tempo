from typing import Any, Callable, List, Optional

from tempo.serve.base import BaseModel
from tempo.serve.constants import ModelDataType
from tempo.serve.metadata import ModelFramework, RuntimeOptions
from tempo.serve.protocol import Protocol
from tempo.serve.runtime import Runtime


class Pipeline(BaseModel):
    def __init__(
        self,
        name: str,
        pipeline_func: Callable[[Any], Any] = None,
        protocol: Optional[Protocol] = None,
        models: List[BaseModel] = None,
        local_folder: str = None,
        uri: str = None,
        inputs: ModelDataType = None,
        outputs: ModelDataType = None,
        conda_env: str = None,
        runtime_options: RuntimeOptions = RuntimeOptions(),
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
            protocol=protocol,
            runtime_options=runtime_options,
        )

        if models is None:
            models = []

        self._models = models

    def set_remote(self, val: bool):
        for model in self._models:
            model.get_tempo().set_remote(val)

    def deploy_models(self, runtime: Runtime):
        for model in self._models:
            model.get_tempo().deploy(runtime)

    def deploy(self, runtime: Runtime):
        self.deploy_models(runtime)
        super().deploy(runtime)

    def wait_ready(self, runtime: Runtime, timeout_secs: int = None) -> bool:
        super().wait_ready(runtime, timeout_secs=timeout_secs)
        for model in self._models:
            if not model.get_tempo().wait_ready(runtime, timeout_secs=timeout_secs):
                return False
        return True

    def undeploy_models(self, runtime: Runtime):
        for model in self._models:
            model.get_tempo().undeploy(runtime)

    def undeploy(self, runtime: Runtime):
        """
        Undeploy all models and pipeline.
        """
        super().undeploy(runtime)
        self.undeploy_models(runtime)

    def to_k8s_yaml(self, runtime: Runtime) -> str:
        yamls = super().to_k8s_yaml(runtime)
        yamls += "\n---\n"
        for model in self._models:
            y = model.get_tempo().to_k8s_yaml(runtime)
            yamls += y
            yamls += "\n---\n"
        return yamls
