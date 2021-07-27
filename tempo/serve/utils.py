import copy
from inspect import getmembers, isclass, isfunction
from types import SimpleNamespace
from typing import Any, Callable, Optional, Type

from ..kfserving.protocol import KFServingV2Protocol
from .base import BaseModel
from .metadata import BaseRuntimeOptionsType, DockerOptions, ModelFramework
from .model import Model
from .pipeline import Pipeline, PipelineModels
from .protocol import Protocol
from .types import ModelDataType

PredictMethodAttr = "_tempo_predict"
LoadMethodAttr = "_tempo_load"


def predictmethod(f):
    setattr(f, PredictMethodAttr, True)
    return f


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


def _get_predict_method_name(K: Type) -> str:
    for name, func in getmembers(K, isfunction):
        if hasattr(func, PredictMethodAttr):
            return name

    return ""


def _wrap_class(K: Type, model: BaseModel, field_name: str = "model") -> Type:
    setattr(K, field_name, model)
    model._K = K

    _bind_tempo_interface(K, model)

    orig_init = K.__init__

    # Make copy of original __init__, so we can call it without recursion
    def __init__(self, *args, **kws):
        # On __init__, copy pipeline object and update _user_func.
        # Class-level attributes mutate to instance-level attributes when
        # overriden.
        # Therefore, this won't modify the class-level model / pipeline object.
        class_model = getattr(K, field_name)
        instance_model = copy.copy(class_model)
        setattr(self, field_name, instance_model)

        # Set predict function name to have same behaviour as __call__
        predictmethod_name = _get_predict_method_name(K)
        setattr(self, predictmethod_name, instance_model.__call__)

        # We bind _user_func so that `self` is passed implicitly
        instance_model._user_func = _bind(self, instance_model._user_func)

        # The copy() function calls __getstate__ so we need to set insights as it's set to SimpleNamespace otheriwse
        setattr(self, "insights_manager", class_model.insights_manager)
        setattr(self, "state", class_model.state)
        setattr(instance_model, "insights_manager", class_model.insights_manager)
        setattr(instance_model, "state", class_model.state)

        # We bind the __getstate__ function to the current object so it also is used when exporting the object
        def __getstate__(self):
            state = self.__dict__.copy()
            state["context"] = SimpleNamespace()
            # Remove the insights manager from the cloudpickle context
            state["insights_manager"] = SimpleNamespace()
            # Remove the state from cloudpickle context
            state["state"] = SimpleNamespace()
            return state

        self.__getstate__ = _bind(self, __getstate__)

        # Bind back Tempo interface to make sure it points to instance referece
        _bind_tempo_interface(self, instance_model)

        orig_init(self, *args, **kws)  # Call the original __init__

    K.__init__ = __init__  # Set the class' __init__ to the new one

    # TODO why not call instance_model._user_func instead if already bound
    # This may be desirable given that it seems there is a discrepancy where the
    # model function gets called with a different self (the self of the model)
    def __call__(self, *args, **kwargs) -> Any:
        model = getattr(self, field_name)
        return model(*args, **kwargs)

    K.__call__ = __call__  # type: ignore

    return K


def _bind_tempo_interface(artifact: Any, model: BaseModel) -> Any:
    setattr(artifact, "request", model.request)
    setattr(artifact, "remote", model.predict)
    setattr(artifact, "get_tempo", model.get_tempo)

    return artifact


def pipeline(
    name: str,
    protocol: Protocol = KFServingV2Protocol(),
    local_folder: str = None,
    uri: str = None,
    models: PipelineModels = None,
    inputs: ModelDataType = None,
    outputs: ModelDataType = None,
    conda_env: str = None,
    runtime_options: BaseRuntimeOptionsType = DockerOptions(),
    description: str = "",
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
    description
     Description of the pipeline

    Returns
    -------
    A decorated class or function.

    """

    def _pipeline(f):
        predict_method = f
        if isclass(f):
            predict_method = _get_predict_method(f)

        pipeline = Pipeline(
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
            description=description,
        )

        if isclass(f):
            K = _wrap_class(f, pipeline, field_name="pipeline")

            @property
            def models_property(self):
                # This is so we do not store reference to `.models` as part of
                # the K class - needed when saving limited copy of models for remote
                return K.pipeline.models

            K.models = models_property

            return K

        return pipeline

    return _pipeline


def model(
    name: str,
    local_folder: str = None,
    uri: str = None,
    platform: ModelFramework = ModelFramework.Custom,
    inputs: ModelDataType = None,
    outputs: ModelDataType = None,
    conda_env: str = None,
    protocol: Protocol = KFServingV2Protocol(),
    runtime_options: BaseRuntimeOptionsType = DockerOptions(),
    description: str = "",
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
    description
     Description of the model

    Returns
    -------
    A decorated function or class as a Tempo Model.

    """

    def _model(f):
        predict_method = f
        if isclass(f):
            predict_method = _get_predict_method(f)

        model = Model(
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
            description=description,
        )

        if isclass(f):
            return _wrap_class(f, model)

        return model

    return _model
