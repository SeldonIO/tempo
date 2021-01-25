from typing import List
from tempo.serve.pipeline import Pipeline
from tempo.serve.runtime import Runtime
from tempo.serve.model import Model
from tempo.serve.metadata import ModelFramework, MetadataTensor

def pipeline(name: str,
            runtime: Runtime = None,
            models: List[Model] = None,
            inputs: List[MetadataTensor] = None,
            outputs: List[MetadataTensor] = None
             ):
    def _pipeline(f):
        # TODO: Infer name from f's name (if not specified)
        return Pipeline(name,runtime=runtime, models=models, inputs=inputs, outputs=outputs, pipeline_func=f)

    return _pipeline
