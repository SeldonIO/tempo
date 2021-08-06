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

    [01;34m.[00m
    ├── [01;34martifacts[00m
    │   ├── [01;34mclassifier[00m
    │   ├── [01;34msklearn[00m
    │   └── [01;34mxgboost[00m
    └── [01;34msrc[00m
        ├── constants.py
        ├── data.py
        ├── tempo.py
        └── train.py
    
    5 directories, 4 files


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
import os

import joblib
from sklearn.linear_model import LogisticRegression
from src.constants import SKLearnFolder, XGBoostFolder
from src.data import IrisData
from xgboost import XGBClassifier


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

    [18:05:52] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.


    /home/clive/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/xgboost/sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


## Create Tempo Artifacts

Here we create the Tempo models and orchestration Pipeline for our final service using our models.
For illustration the final service will call the sklearn model and based on the result will decide to return that prediction or call the xgboost model and return that prediction instead.


```python
from src.tempo import classifier
```


```python
# %load src/tempo.py
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

```

## Save Classifier Environment

In preparation for running our models we save the Python environment needed for the orchestration to run as defined by a `conda.yaml` in our project.


```python
!ls artifacts/classifier/conda.yaml
```

    artifacts/classifier/conda.yaml



```python
import tempo

tempo.save(classifier)
```

    Collecting packages...
    Packing environment at '/home/clive/anaconda3/envs/tempo-330c15d8-a189-45a6-abc3-a27f39b6a7c5' to '/home/clive/work/mlops/fork-tempo/docs/examples/asyncio/artifacts/classifier/environment.tar.gz'
    [########################################] | 100% Completed | 11.2s


## Test Locally on Docker

Here we test our models using production images but running locally on Docker. This allows us to ensure the final production deployed model will behave as expected when deployed.


```python
from tempo.aio import deploy_local
remote_model = deploy_local(classifier)
```


```python
import numpy as np
await remote_model.predict(np.array([[1, 2, 3, 4]]))
```




    array([2.], dtype=float32)




```python
print(await remote_model.predict(np.array([[0, 0, 0,0]])))
print(await remote_model.predict(np.array([[5.964,4.006,2.081,1.031]])))
```

    [1.]
    [[0.97329617 0.02412145 0.00258233]]



```python
remote_model.undeploy()
```


```python

```
