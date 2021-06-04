import numpy as np

from tempo import ModelFramework, PipelineModels
from tempo.aio import pipeline, Model

from src.constants import SKLearnFolder, XGBoostFolder

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
    name="inference-pipeline",
    models=PipelineModels(sklearn=SKLearnModel, xgboost=XGBoostModel),
)
async def inference_pipeline(payload: np.ndarray) -> np.ndarray:
    res1 = await _pipeline.models.sklearn(input=payload)
    if res1[0][0] > 0.7:
        return res1

    return await _pipeline.models.xgboost(input=payload)
