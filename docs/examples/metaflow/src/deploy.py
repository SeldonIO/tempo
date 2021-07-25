from typing import Tuple

import numpy as np

from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.serve.pipeline import Pipeline, PipelineModels
from tempo.serve.utils import pipeline

PipelineFolder = "classifier"
SKLearnTag = "sklearn prediction"
XGBoostTag = "xgboost prediction"


def get_tempo_artifacts(
    sklearn_local_path: str, sklearn_uri: str, xgboost_local_path: str, xgboost_uri: str, classifier_uri: str
) -> Tuple[Pipeline, Model, Model]:

    sklearn_model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        local_folder=sklearn_local_path,
        uri=sklearn_uri,
        description="An SKLearn Iris classification model",
    )

    xgboost_model = Model(
        name="test-iris-xgboost",
        platform=ModelFramework.XGBoost,
        local_folder=xgboost_local_path,
        uri=xgboost_uri,
        description="An XGBoost Iris classification model",
    )

    @pipeline(
        name="classifier",
        uri=classifier_uri,
        models=PipelineModels(sklearn=sklearn_model, xgboost=xgboost_model),
        description="A pipeline to use either an sklearn or xgboost model for Iris classification",
    )
    def classifier(payload: np.ndarray) -> Tuple[np.ndarray, str]:
        res1 = classifier.models.sklearn(input=payload)

        if res1[0] == 1:
            return res1, SKLearnTag
        else:
            return classifier.models.xgboost(input=payload), XGBoostTag

    return classifier, sklearn_model, xgboost_model
