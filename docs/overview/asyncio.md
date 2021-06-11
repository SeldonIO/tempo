# AsyncIO support in Tempo

Tempo includes experimental support to write concurrent code using [Python's
AsyncIO](https://docs.python.org/3/library/asyncio.html).
Using AsyncIO can be beneficial in scenarios where most of the heavy lifting is
done by downstream models and the pipeline just orchestrates calls across these
models.
In this case, most of the time within the pipeline will be spent waiting for
the requests from downstream models to come back.
AsyncIO will allow us to process other incoming requests during this waiting
time.

## Usage

To use AsyncIO in your Tempo models and pipelines, you only need to import
Tempo's interfaces from the `tempo.aio` package (i.e. instead of just `tempo`).
For example, to replicate the pipeline shown in [Tempo's
overview](./overview.md), you could do the following:

```python
import numpy as np

from tempo import ModelFramework
from tempo.aio import pipeline, Model, PipelineModels

from src.constants import ClassifierFolder, SKLearnFolder, XGBoostFolder

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
```

## Example

For more details, check out [this worked out example](../examples/asyncio/README.md).
