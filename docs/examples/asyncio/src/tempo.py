import numpy as np
from src.constants import ClassifierFolder, SKLearnFolder, XGBoostFolder

from tempo import ModelFramework, PipelineModels
from tempo.aio import Model, pipeline

SKLearnModel = Model(
    name="test-iris-sklearn",
    platform=ModelFramework.SKLearn,
    local_folder=SKLearnFolder,
    uri="s3://tempo/basic/sklearn",
    description="An SKLearn Iris classification model",
)

XGBoostModel = Model(
    name="test-iris-xgboost",
    platform=ModelFramework.XGBoost,
    local_folder=XGBoostFolder,
    uri="s3://tempo/basic/xgboost",
    description="An XGBoost Iris classification model",
)


@pipeline(
    name="classifier",
    models=PipelineModels(sklearn=SKLearnModel, xgboost=XGBoostModel),
    local_folder=ClassifierFolder,
)
async def classifier(payload: np.ndarray) -> np.ndarray:
    res1 = await classifier.models.sklearn(input=payload)
    if res1[0] > 0.7:
        return res1

    return await classifier.models.xgboost(input=payload)
