from inspect import isclass

from ..kfserving.protocol import KFServingV2Protocol
from ..serve.metadata import ModelFramework, RuntimeOptions
from ..serve.pipeline import PipelineModels
from ..serve.protocol import Protocol
from ..serve.types import ModelDataType
from ..serve.utils import _get_predict_method, _wrap_class
from .model import Model
from .pipeline import Pipeline


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
    runtime_options: RuntimeOptions = RuntimeOptions(),
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
