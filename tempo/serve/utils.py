import inspect
from inspect import getmembers, isfunction
from typing import Any, Callable, Optional, Type

from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.metadata import ModelFramework, RuntimeOptions
from tempo.serve.model import Model
from tempo.serve.pipeline import Pipeline, PipelineModels
from tempo.serve.protocol import Protocol
from tempo.serve.types import ModelDataType

PredictMethodAttr = "_tempo_predict"
LoadMethodAttr = "_tempo_load"


def _bind(instance, func):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".

    From https://stackoverflow.com/a/1015405/5015573
    """
    return func.__get__(instance, instance.__class__)


def _get_predict_method(K: Type) -> Optional[Callable]:
    for _, func in getmembers(K, isfunction):
        if hasattr(func, PredictMethodAttr):
            return func

    return None


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
    """
    A decorator for a class or function to make it a Tempo Pipeline.

    Parameters
    ----------
    name
     Name of the pipeline. Needs to be Kubernetes compliant.
    protocol
     :class:`tempo.serve.protocol.Protocol`. Defaults to KFserving V2.
    local_folder
     Location of local artifacts.
    uri
     Location of remote artifacts.
    models
     A list of models defined as PipelineModels.
    inputs
     The input types.
    outputs
     The output types.
    conda_env
     The conda environment name to use. If not specified will look for conda.yaml in local_folder
     or generate from current running environment.
    runtime_options
     The runtime options. Can be left empty and set when creating a runtime.

    Returns
    -------
    A decorated class or function.

    """

    def _pipeline(f):
        if inspect.isclass(f):
            K = f
            predict_method = _get_predict_method(K)

            K.pipeline = Pipeline(
                name,
                local_folder=local_folder,
                uri=uri,
                models=models,
                inputs=inputs,
                outputs=outputs,
                pipeline_func=predict_method,
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
                # We bind _user_func so that `self` is passed implicitly
                K.pipeline._user_func = _bind(self, K.pipeline._user_func)
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
    setattr(f, PredictMethodAttr, True)
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
    """

    Parameters
    ----------
    name
     Name of the model. Needs to be Kubernetes compliant.
    protocol
     :class:`tempo.serve.protocol.Protocol`. Defaults to KFserving V2.
    local_folder
     Location of local artifacts.
    uri
     Location of remote artifacts.
    inputs
     The input types.
    outputs
     The output types.
    conda_env
     The conda environment name to use. If not specified will look for conda.yaml in local_folder
     or generate from current running environment.
    runtime_options
     The runtime options. Can be left empty and set when creating a runtime.
    platform
     The :class:`tempo.serve.metadata.ModelFramework`

    Returns
    -------
    A decorated function or class as a Tempo Model.

    """

    def _model(f):
        if inspect.isclass(f):
            K = f
            predict_method = _get_predict_method(K)

            K.pipeline = Model(
                name,
                protocol=protocol,
                local_folder=local_folder,
                uri=uri,
                platform=platform,
                inputs=inputs,
                outputs=outputs,
                model_func=predict_method,
                conda_env=conda_env,
                runtime_options=runtime_options,
            )

            setattr(K, "request", K.pipeline.request)
            setattr(K, "remote", K.pipeline.remote)
            setattr(K, "get_tempo", K.pipeline.get_tempo)

            orig_init = K.__init__

            # Make copy of original __init__, so we can call it without recursion
            def __init__(self, *args, **kws):
                # We bind _user_func so that `self` is passed implicitly
                K.pipeline._user_func = _bind(self, K.pipeline._user_func)
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
