import inspect
from typing import Any, List

from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.constants import ModelDataType
from tempo.serve.metadata import ModelFramework, RuntimeOptions
from tempo.serve.model import Model
from tempo.serve.pipeline import PipelineModels
from tempo.serve.pipeline import Pipeline
from tempo.serve.protocol import Protocol


def pipeline(
    name: str,
    protocol: Protocol = KFServingV2Protocol(),
    local_folder: str = None,
    uri: str = None,
    models: PipelineModels = None,
    inputs: ModelDataType = None,
    outputs: ModelDataType = None,
    conda_env: str = None,
    runtime_options: RuntimeOptions = RuntimeOptions(),
):
    def _pipeline(f):
        if inspect.isclass(f):
            K = f
            func = None

            for a in dir(K):
                if not a.startswith("__") and callable(getattr(K, a)) and hasattr(getattr(K, a), "predict"):
                    func = getattr(K, a)
                    break
            K.pipeline = Pipeline(
                name,
                local_folder=local_folder,
                uri=uri,
                models=models,
                inputs=inputs,
                outputs=outputs,
                pipeline_func=func,
                conda_env=conda_env,
                protocol=protocol,
                runtime_options=runtime_options,
            )
            setattr(K, "request", K.pipeline.request)
            setattr(K, "remote", K.pipeline.remote)
            setattr(K, "get_tempo", K.pipeline.get_tempo)

            orig_init = K.__init__

            # Make copy of original __init__, so we can call it without recursion
            def __init__(self, *args, **kws):
                K.pipeline.set_cls(self)
                orig_init(self, *args, **kws)  # Call the original __init__

            K.__init__ = __init__  # Set the class' __init__ to the new one

            def __call__(self, *args, **kwargs) -> Any:
                return self.pipeline(*args, **kwargs)

            K.__call__ = __call__

            @property
            def models_property(self):
                # This is so we do not store reference to `.models` as part of
                # the K class - needed when saving limited copy of models for remote
                return K.pipeline.models

            K.models = models_property

            return K
        else:
            return Pipeline(
                name,
                local_folder=local_folder,
                uri=uri,
                models=models,
                inputs=inputs,
                outputs=outputs,
                pipeline_func=f,
                conda_env=conda_env,
                protocol=protocol,
                runtime_options=runtime_options,
            )

    return _pipeline


def predictmethod(f):
    f.predict = True
    return f


def model(
    name: str,
    local_folder: str = None,
    uri: str = None,
    platform: ModelFramework = None,
    inputs: ModelDataType = None,
    outputs: ModelDataType = None,
    conda_env: str = None,
    protocol: Protocol = KFServingV2Protocol(),
    runtime_options: RuntimeOptions = RuntimeOptions(),
):
    def _model(f):
        if inspect.isclass(f):
            K = f
            func = None

            for a in dir(K):
                if not a.startswith("__") and callable(getattr(K, a)) and hasattr(getattr(K, a), "predict"):
                    func = getattr(K, a)
                    break

            K.pipeline = Model(
                name,
                protocol=protocol,
                local_folder=local_folder,
                uri=uri,
                platform=platform,
                inputs=inputs,
                outputs=outputs,
                model_func=func,
                conda_env=conda_env,
                runtime_options=runtime_options,
            )

            setattr(K, "request", K.pipeline.request)
            setattr(K, "remote", K.pipeline.remote)
            setattr(K, "get_tempo", K.pipeline.get_tempo)

            orig_init = K.__init__

            # Make copy of original __init__, so we can call it without recursion
            def __init__(self, *args, **kws):
                K.pipeline.set_cls(self)
                orig_init(self, *args, **kws)  # Call the original __init__

            K.__init__ = __init__  # Set the class' __init__ to the new one

            def __call__(self, *args, **kwargs) -> Any:
                return self.pipeline(*args, **kwargs)

            K.__call__ = __call__

            return K
        else:
            return Model(
                name,
                protocol=protocol,
                local_folder=local_folder,
                uri=uri,
                platform=platform,
                inputs=inputs,
                outputs=outputs,
                model_func=f,
                conda_env=conda_env,
                runtime_options=runtime_options,
            )

    return _model
