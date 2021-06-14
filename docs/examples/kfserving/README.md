# Deploy to KFserving

![architecture](architecture.png)

In this introduction we will:

  * [Describe the project structure](#Project-Structure)
  * [Train some models](#Train-Models)
  * [Create Tempo artifacts](#Create-Tempo-Artifacts)
  * [Run unit tests](#Unit-Tests)
  * [Save python environment for our classifier](#Save-Classifier-Environment)
  * [Test Locally on Docker](#Test-Locally-on-Docker)
  * [Production on Kubernetes via Tempo](#Production-Option-1-(Deploy-to-Kubernetes-with-Tempo))
  * [Prodiuction on Kuebrnetes via GitOps](#Production-Option-2-(Gitops))

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

 * This section is where as a data scientist you do your work of training models and creating artfacts.
 * For this example we train sklearn and xgboost classification models for the iris dataset.


```python
import os
from tempo.utils import logger
import logging
import numpy as np
logger.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
ARTIFACTS_FOLDER = os.getcwd()+"/artifacts"
```


```python
# %load src/train.py
from typing import Tuple

import joblib
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SKLearnFolder = "sklearn"
XGBoostFolder = "xgboost"


def load_iris() -> Tuple[np.ndarray, np.ndarray]:
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    return (X, y)


def train_sklearn(X: np.ndarray, y: np.ndarray, artifacts_folder: str):
    logreg = LogisticRegression(C=1e5)
    logreg.fit(X, y)
    logreg.predict_proba(X[0:1])
    with open(f"{artifacts_folder}/{SKLearnFolder}/model.joblib", "wb") as f:
        joblib.dump(logreg, f)


def train_xgboost(X: np.ndarray, y: np.ndarray, artifacts_folder: str):
    clf = XGBClassifier()
    clf.fit(X, y)
    clf.save_model(f"{artifacts_folder}/{XGBoostFolder}/model.bst")

```


```python
from src.data import IrisData
from src.train import train_lr, train_xgb
data = IrisData()

train_lr(ARTIFACTS_FOLDER, data)
train_xgb(ARTIFACTS_FOLDER, data)
```

## Create Tempo Artifacts



```python
from src.tempo import get_tempo_artifacts
classifier, sklearn_model, xgboost_model = get_tempo_artifacts(ARTIFACTS_FOLDER)
```


```python
# %load src/tempo.py
from typing import Tuple

import numpy as np
from src.constants import SKLearnFolder, XGBFolder, SKLearnTag, XGBoostTag

from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.serve.pipeline import Pipeline, PipelineModels
from tempo.serve.utils import pipeline


def get_tempo_artifacts(artifacts_folder: str) -> Tuple[Pipeline, Model, Model]:
    sklearn_model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        local_folder=f"{artifacts_folder}/{SKLearnFolder}",
        uri="s3://tempo/basic/sklearn",
        description="SKLearn Iris classification model",
    )

    xgboost_model = Model(
        name="test-iris-xgboost",
        platform=ModelFramework.XGBoost,
        local_folder=f"{artifacts_folder}/{XGBFolder}",
        uri="s3://tempo/basic/xgboost",
        description="XGBoost Iris classification model",
    )

    @pipeline(
        name="classifier",
        uri="s3://tempo/basic/pipeline",
        local_folder=f"{artifacts_folder}/classifier",
        models=PipelineModels(sklearn=sklearn_model, xgboost=xgboost_model),
        description="A pipeline to use either an sklearn or xgboost model for Iris classification",
    )
    def classifier(payload: np.ndarray) -> Tuple[np.ndarray, str]:
        res1 = classifier.models.sklearn(input=payload)
        print(res1)
        if res1[0] == 1:
            return res1, SKLearnTag
        else:
            return classifier.models.xgboost(input=payload), XGBoostTag

    return classifier, sklearn_model, xgboost_model

```

## Unit Tests

 * Here we run our unit tests to ensure the orchestration works before running on the actual models.


```python
# %load tests/test_deploy.py
import numpy as np
from src.tempo import get_tempo_artifacts
from src.constants import SKLearnTag, XGBoostTag


def test_sklearn_model_used():
    classifier, _, _ = get_tempo_artifacts("")
    classifier.models.sklearn = lambda input: np.array([[1]])
    res, tag = classifier(np.array([[1, 2, 3, 4]]))
    assert res[0][0] == 1
    assert tag == SKLearnTag


def test_xgboost_model_used():
    classifier, _, _ = get_tempo_artifacts("")
    classifier.models.sklearn = lambda input: np.array([[0.2]])
    classifier.models.xgboost = lambda input: np.array([[0.1]])
    res, tag = classifier(np.array([[1, 2, 3, 4]]))
    assert res[0][0] == 0.1
    assert tag == XGBoostTag

```


```python
!python -m pytest tests/
```

## Save Classifier Environment

 * In preparation for running our models we save the Python environment needed for the orchestration to run as defined by a `conda.yaml` in our project.


```python
!cat artifacts/classifier/conda.yaml
```


```python
from tempo.serve.loader import save
save(classifier)
```

## Test Locally on Docker

 * Here we test our models using production images but running locally on Docker. This allows us to ensure the final production deployed model will behave as expected when deployed.


```python
from tempo import deploy
remote_model = deploy(classifier)
```


```python
print(remote_model.predict(np.array([[0, 0, 0,0]])))
print(remote_model.predict(np.array([[5.964,4.006,2.081,1.031]])))
```


```python
remote_model.undeploy()
```

## Production Option 1 (Deploy to Kubernetes with Tempo)

 * Here we illustrate how to run the final models in "production" on Kubernetes by using Tempo to deploy
 
### Prerequisites
 
Create a Kind Kubernetes cluster with Minio and KFserving installed using Ansible as described [here](../../overview/quickstart.md#kubernetes-cluster-with-kfserving).


```python
!kubectl create ns production
```


```python
!kubectl apply -f k8s/rbac -n production
```


```python
from tempo.examples.minio import create_minio_rclone
import os
create_minio_rclone(os.getcwd()+"/rclone.conf")
```


```python
from tempo.serve.loader import upload
upload(sklearn_model)
upload(xgboost_model)
upload(classifier)
```


```python
from tempo.serve.metadata import KubernetesOptions
from tempo.kfserving.k8s import KFServingOptions
runtime_options = KFServingOptions(
        k8s_options=KubernetesOptions(
            runtime="tempo.kfserving.KFServingKubernetesRuntime",
            namespace="production",
            serviceAccountName="kf-tempo"
        )
    )
```


```python
from tempo import deploy
remote_model = deploy(classifier, options=runtime_options)
```


```python
print(remote_model.predict(payload=np.array([[0, 0, 0, 0]])))
print(remote_model.predict(payload=np.array([[1, 2, 3, 4]])))
```

### Illustrate client using model remotely

With the Kubernetes runtime one can list running models on the Kubernetes cluster and instantiate a RemoteModel to call the Tempo model.


```python
from tempo.kfserving.k8s import KFServingKubernetesRuntime
k8s_runtime = KFServingKubernetesRuntime(runtime_options)
models = k8s_runtime.list_models(namespace="production")
print("Name\tDescription")
for model in models:
    details = model.get_tempo().model_spec.model_details
    print(f"{details.name}\t{details.description}")
```


```python
models[0].predict(payload=np.array([[1, 2, 3, 4]]))
```


```python
remote_model.undeploy()
```

## Production Option 2 (Gitops)

 * We create yaml to provide to our DevOps team to deploy to a production cluster
 * We add Kustomize patches to modify the base Kubernetes yaml created by Tempo


```python
from tempo.kfserving.k8s import KFServingKubernetesRuntime
k8s_runtime = KFServingKubernetesRuntime(runtime_options)
yaml_str = k8s_runtime.manifest(classifier)
with open(os.getcwd()+"/k8s/tempo.yaml","w") as f:
    f.write(yaml_str)
```


```python
!kustomize build k8s
```


```python

```
