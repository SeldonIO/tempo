from typing import List, Optional, Dict, Type, Union
from tempo.serve.constants import ModelDataType
from tempo.serve.pipeline import Pipeline
from tempo.serve.runtime import Runtime
from tempo.serve.model import Model
from tempo.serve.metadata import ModelFramework

def pipeline(name: str,
            runtime: Runtime = None,
            models: List[Model] = None,
            inputs: ModelDataType = None,
            outputs: ModelDataType = None
             ):
    def _pipeline(f):
        return Pipeline(name,runtime=runtime, models=models, inputs=inputs, outputs=outputs, pipeline_func=f)

    return _pipeline


def model(name: str,
          runtime: Runtime = None,
          local_folder: str = None,
          uri: str = None,
          platform: ModelFramework = None,
          inputs: ModelDataType = None,
          outputs: ModelDataType = None
          ):
    def _model(f):
        return Model(name, runtime=runtime, local_folder=local_folder,uri=uri,platform=platform,inputs=inputs,outputs=outputs, model_func=f)

    return _model