from typing import List, Optional, Dict, Type, Union
from tempo.serve.constants import ModelDataType
from tempo.serve.pipeline import Pipeline
from tempo.serve.runtime import Runtime
from tempo.serve.model import Model
from tempo.serve.metadata import ModelFramework
import inspect


def pipeline(
    name: str,
    runtime: Runtime = None,
    local_folder: str = None,
    uri: str = None,
    models: List[Model] = None,
    inputs: ModelDataType = None,
    outputs: ModelDataType = None,
):
    def _pipeline(f):
        if inspect.isclass(f):
            K = f
            func = None

            for a in dir(K):
                if (
                    not a.startswith("__")
                    and callable(getattr(K, a))
                    and hasattr(getattr(K, a), "predict")
                ):
                    func = getattr(K, a)
                    break
            K.pipeline = Pipeline(
                name,
                runtime=runtime,
                local_folder=local_folder,
                uri=uri,
                models=models,
                inputs=inputs,
                outputs=outputs,
                pipeline_func=func,
            )
            setattr(K, "deploy", K.pipeline.deploy)
            setattr(K, "deploy_models", K.pipeline.deploy_models)
            setattr(K, "wait_ready", K.pipeline.wait_ready)
            setattr(K, "undeploy", K.pipeline.undeploy)
            setattr(K, "undeploy_models", K.pipeline.undeploy_models)
            setattr(K, "request", K.pipeline.request)
            setattr(K, "set_runtime", K.pipeline.set_runtime)
            setattr(K, "to_k8s_yaml", K.pipeline.to_k8s_yaml)

            orig_init = K.__init__
            # Make copy of original __init__, so we can call it without recursion
            def __init__(self, *args, **kws):
                K.pipeline.set_cls(self)
                orig_init(self, *args, **kws)  # Call the original __init__

            K.__init__ = __init__  # Set the class' __init__ to the new one

            return K
        else:
            return Pipeline(
                name,
                runtime=runtime,
                local_folder=local_folder,
                uri=uri,
                models=models,
                inputs=inputs,
                outputs=outputs,
                pipeline_func=f,
            )

    return _pipeline


def predictmethod(f):
    f.predict = True
    return f


def model(
    name: str,
    runtime: Runtime = None,
    local_folder: str = None,
    uri: str = None,
    platform: ModelFramework = None,
    inputs: ModelDataType = None,
    outputs: ModelDataType = None,
):
    def _model(f):
        return Model(
            name,
            runtime=runtime,
            local_folder=local_folder,
            uri=uri,
            platform=platform,
            inputs=inputs,
            outputs=outputs,
            model_func=f,
        )

    return _model
