from inspect import isclass

from ..kfserving.protocol import KFServingV2Protocol

from ..serve.utils import _get_predict_method, _wrap_class
from ..serve.metadata import ModelFramework, RuntimeOptions
from ..serve.types import ModelDataType
from ..serve.protocol import Protocol

from .model import Model


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
