# Leveraging AsyncIO in inference pipelines

Tempo includes experimental support for `asyncio`, which provides a way to optimise pipelines.
In particular, `asyncio` can be beneficial in scenarios where most of the heavy lifting is done by downstream models and the pipeline just orchestrates calls across these models.
In this case, most of the time within the pipeline will be spent waiting for the requests from downstream models to come back.
`asyncio` will allow us to process other incoming requests during this waiting time.

This example will walk us through the process of setting up an asynchronous pipeline.
As you will see, it's quite similar to the usual synchronous pipelines.

## Prerequisites

This notebooks needs to be run in the `tempo-examples` conda environment defined below. Create from project root folder:

```bash
conda env create --name tempo-examples --file conda/tempo-examples.yaml
```

## Project Structure


```python
!tree -P "*.py"  -I "__init__.py|__pycache__" -L 2
```

## Train Models

This section is where as a data scientist you do your work of training models and creating artfacts.
For this example, we will train two sklearn and xgboost classification models using the iris dataset.

These models will be used by our inference pipeline.


```python
import logging
from tempo.utils import logger

logger.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
```


```python
# %load src/train.py
import joblib
import os

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data import IrisData
from src.constants import SKLearnFolder, XGBoostFolder


def train_sklearn(data: IrisData):
    logreg = LogisticRegression(C=1e5)
    logreg.fit(data.X, data.y)

    model_path = os.path.join(SKLearnFolder, "model.joblib")
    with open(model_path, "wb") as f:
        joblib.dump(logreg, f)


def train_xgboost(data: IrisData):
    clf = XGBClassifier()
    clf.fit(data.X, data.y)

    model_path = os.path.join(XGBoostFolder, "model.json")
    clf.save_model(model_path)

```


```python
from src.data import IrisData
from src.train import train_sklearn, train_xgboost

data = IrisData()

train_sklearn(data)
train_xgboost(data)
```

## Create Tempo Artifacts

Here we create the Tempo models and orchestration Pipeline for our final service using our models.
For illustration the final service will call the sklearn model and based on the result will decide to return that prediction or call the xgboost model and return that prediction instead.


```python
from src.tempo import classifier
```


```python
# %load src/tempo.py
import numpy as np

from tempo import ModelFramework, PipelineModels
from tempo.aio import pipeline, Model

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

## Save Classifier Environment

In preparation for running our models we save the Python environment needed for the orchestration to run as defined by a `conda.yaml` in our project.


```python
!cat artifacts/classifier/conda.yaml
```


```python
import tempo

tempo.save(classifier)
```

## Test Locally on Docker

Here we test our models using production images but running locally on Docker. This allows us to ensure the final production deployed model will behave as expected when deployed.


```python
from tempo.seldon.docker import SeldonDockerRuntime

docker_runtime = SeldonDockerRuntime()
docker_runtime.deploy(classifier)
docker_runtime.wait_ready(classifier)
```


```python
await classifier(np.array([[1, 2, 3, 4]]))
```


```python
print(await classifier.remote(np.array([[0, 0, 0,0]])))
print(await classifier.remote(np.array([[5.964,4.006,2.081,1.031]])))
```


```python
docker_runtime.undeploy(classifier)
```


```python

```
