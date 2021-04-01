from types import SimpleNamespace
from typing import Any, Callable, Optional

from tempo.serve.base import BaseModel
from tempo.serve.metadata import ModelFramework, RuntimeOptions
from tempo.serve.model import Model
from tempo.serve.protocol import Protocol
from tempo.serve.runtime import Runtime
from tempo.serve.types import ModelDataType


class PipelineModels(SimpleNamespace):
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def remote_copy(self):
        output = {}
        for name, model in self.items():
            output[name] = Model(
                name=model.get_tempo().details.name,
                local_folder=model.get_tempo().details.local_folder,
                uri=model.get_tempo().details.uri,
                platform=model.get_tempo().details.platform,
                inputs=model.get_tempo().details.inputs,
                outputs=model.get_tempo().details.outputs,
                protocol=model.get_tempo().protocol,
                runtime_options=model.get_tempo().model_spec.runtime_options,
            )
        return PipelineModels(**output)


class Pipeline(BaseModel):
    def __init__(
        self,
        name: str,
        pipeline_func: Callable[[Any], Any] = None,
        protocol: Optional[Protocol] = None,
        models: PipelineModels = None,
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
            models = PipelineModels()

        self.models = models

    def save(self, save_env=True):
        # Backup models and create lite copy so Pipeline doesn not inherit dependencies
        models_backup = self.models
        self.models = self.models.remote_copy()
        super().save(save_env=save_env)

        # Restore backed up models
        self.models = models_backup

    def set_remote(self, val: bool):
        for model in self.models.values():
            model.get_tempo().set_remote(val)

    def set_runtime_options_override(self, runtime_options: RuntimeOptions):
        for model in self.models.values():
            model.get_tempo().set_runtime_options_override(runtime_options)
        super().set_runtime_options_override(runtime_options)

    def deploy_models(self, runtime: Runtime):
        for model in self.models.values():
            model.get_tempo().deploy(runtime)

    def deploy(self, runtime: Runtime):
        self.deploy_models(runtime)
        super().deploy(runtime)

    def wait_ready(self, runtime: Runtime, timeout_secs: int = None) -> bool:
        super().wait_ready(runtime, timeout_secs=timeout_secs)
        for model in self.models.values():
            if not model.get_tempo().wait_ready(runtime, timeout_secs=timeout_secs):
                return False
        return True

    def undeploy_models(self, runtime: Runtime):
        for model in self.models.values():
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
        for model in self.models.values():
            y = model.get_tempo().to_k8s_yaml(runtime)
            yamls += y
            yamls += "\n---\n"
        return yamls
