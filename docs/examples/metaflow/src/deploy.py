import tempfile
from typing import Tuple

import numpy as np
from metaflow import FlowSpec, IncludeFile

from tempo.metaflow.utils import create_s3_folder, save_pipeline_with_conda, upload_s3_folder
from tempo.serve.model import Model
from tempo.serve.pipeline import Pipeline, PipelineModels
from tempo.serve.utils import pipeline

PipelineFolder = "classifier"


def get_tempo_artifacts(
    flow_spec: FlowSpec, sklearn_model: Model, xgboost_model: Model, conda_env_path: IncludeFile
) -> Tuple[Pipeline, bool]:

    classifier_local_path = tempfile.mkdtemp()
    classifier_url = create_s3_folder(flow_spec, PipelineFolder)

    @pipeline(
        name="classifier",
        uri=classifier_url,
        local_folder=classifier_local_path,
        models=PipelineModels(sklearn=sklearn_model, xgboost=xgboost_model),
        description="A pipeline to use either an sklearn or xgboost model for Iris classification",
    )
    def classifier(payload: np.ndarray) -> Tuple[np.ndarray, str]:
        res1 = classifier.models.sklearn(input=payload)

        if res1[0] == 1:
            return res1, "sklearn prediction"
        else:
            return classifier.models.xgboost(input=payload), "xgboost prediction"

    save_pipeline_with_conda(classifier, classifier_local_path, conda_env_path)
    if classifier_url:
        upload_s3_folder(flow_spec, PipelineFolder, classifier_local_path)

    return classifier, classifier_url != ""
