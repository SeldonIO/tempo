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

    [01;34m.[00m
    ├── [01;34martifacts[00m
    │   ├── [01;34mclassifier[00m
    │   ├── [01;34msklearn[00m
    │   └── [01;34mxgboost[00m
    ├── [01;34mk8s[00m
    │   └── [01;34mrbac[00m
    ├── [01;34msrc[00m
    │   ├── constants.py
    │   ├── data.py
    │   ├── tempo.py
    │   └── train.py
    └── [01;34mtests[00m
        └── test_deploy.py
    
    8 directories, 5 files


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

    [11:51:11] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.


    /home/clive/anaconda3/envs/tempo-dev/lib/python3.7/site-packages/xgboost/sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


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

    [1m============================= test session starts ==============================[0m
    platform linux -- Python 3.7.10, pytest-6.2.0, py-1.10.0, pluggy-0.13.1
    rootdir: /home/clive/work/mlops/fork-tempo, configfile: setup.cfg
    plugins: cases-3.4.6, asyncio-0.14.0
    collected 2 items                                                              [0m[1m
    
    tests/test_deploy.py [32m.[0m[32m.[0m[32m                                                  [100%][0m
    
    [32m============================== [32m[1m2 passed[0m[32m in 0.75s[0m[32m ===============================[0m


## Save Classifier Environment

 * In preparation for running our models we save the Python environment needed for the orchestration to run as defined by a `conda.yaml` in our project.


```python
!ls artifacts/classifier/conda.yaml
```

    artifacts/classifier/conda.yaml



```python
from tempo.serve.loader import save
save(classifier)
```

    Collecting packages...
    Packing environment at '/home/clive/anaconda3/envs/tempo-b078d4e0-48a7-4c6e-bf46-74fc623ea46a' to '/home/clive/work/mlops/fork-tempo/docs/examples/kfserving/artifacts/classifier/environment.tar.gz'
    [########################################] | 100% Completed | 13.7s


## Test Locally on Docker

 * Here we test our models using production images but running locally on Docker. This allows us to ensure the final production deployed model will behave as expected when deployed.


```python
from tempo import deploy_local
remote_model = deploy_local(classifier)
```


```python
print(remote_model.predict(np.array([[0, 0, 0,0]])))
print(remote_model.predict(np.array([[5.964,4.006,2.081,1.031]])))
```

    {'output0': array([1.], dtype=float32), 'output1': 'sklearn prediction'}
    {'output0': array([[0.97329617, 0.02412145, 0.00258233]], dtype=float32), 'output1': 'xgboost prediction'}



```python
remote_model.undeploy()
```

## Production Option 1 (Deploy to Kubernetes with Tempo)

 * Here we illustrate how to run the final models in "production" on Kubernetes by using Tempo to deploy
 
### Prerequisites
 
Create a Kind Kubernetes cluster with Minio and KFserving installed using Ansible as described [here](https://tempo.readthedocs.io/en/latest/overview/quickstart.html#kubernetes-cluster-with-kfserving).


```python
!kubectl create ns production
```

    Error from server (AlreadyExists): namespaces "production" already exists



```python
!kubectl apply -f k8s/rbac -n production
```

    secret/minio-secret configured
    serviceaccount/kf-tempo configured
    role.rbac.authorization.k8s.io/kf-tempo unchanged
    rolebinding.rbac.authorization.k8s.io/tempo-pipeline-rolebinding unchanged



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
from tempo.serve.metadata import SeldonCoreOptions
runtime_options = SeldonCoreOptions(**{
        "remote_options": {
            "runtime": "tempo.kfserving.KFServingKubernetesRuntime",
            "namespace": "production",
            "serviceAccountName": "kf-tempo"
        }
    })
```


```python
from tempo import deploy_remote
remote_model = deploy_remote(classifier, options=runtime_options)
```


```python
print(remote_model.predict(payload=np.array([[0, 0, 0, 0]])))
print(remote_model.predict(payload=np.array([[1, 2, 3, 4]])))
```

    {'output0': array([1.], dtype=float32), 'output1': 'sklearn prediction'}
    {'output0': array([[0.00847207, 0.03168794, 0.95984   ]], dtype=float32), 'output1': 'xgboost prediction'}


### Illustrate client using model remotely

With the Kubernetes runtime one can list running models on the Kubernetes cluster and instantiate a RemoteModel to call the Tempo model.


```python
from tempo.kfserving.k8s import KFServingKubernetesRuntime
k8s_runtime = KFServingKubernetesRuntime(runtime_options.remote_options)
models = k8s_runtime.list_models(namespace="production")
print("Name\tDescription")
for model in models:
    details = model.get_tempo().model_spec.model_details
    print(f"{details.name}\t{details.description}")
```

    Name	Description
    classifier	A pipeline to use either an sklearn or xgboost model for Iris classification
    test-iris-sklearn	SKLearn Iris classification model
    test-iris-xgboost	XGBoost Iris classification model



```python
models[0].predict(payload=np.array([[1, 2, 3, 4]]))
```




    {'output0': array([[0.00847207, 0.03168794, 0.95984   ]], dtype=float32),
     'output1': 'xgboost prediction'}




```python
remote_model.undeploy()
```

## Production Option 2 (Gitops)

 * We create yaml to provide to our DevOps team to deploy to a production cluster
 * We add Kustomize patches to modify the base Kubernetes yaml created by Tempo


```python
from tempo import manifest
from tempo.serve.metadata import SeldonCoreOptions
runtime_options = SeldonCoreOptions(**{
        "remote_options": {
            "runtime": "tempo.kfserving.KFServingKubernetesRuntime",
            "namespace": "production",
            "serviceAccountName": "kf-tempo"
        }
    })
yaml_str = manifest(classifier, options=runtime_options)
with open(os.getcwd()+"/k8s/tempo.yaml","w") as f:
    f.write(yaml_str)
```


```python
!kustomize build k8s
```

    apiVersion: serving.kubeflow.org/v1beta1
    kind: InferenceService
    metadata:
      annotations:
        seldon.io/tempo-description: A pipeline to use either an sklearn or xgboost model
          for Iris classification
        seldon.io/tempo-model: '{"model_details": {"name": "classifier", "local_folder":
          "/home/clive/work/mlops/fork-tempo/docs/examples/kfserving/artifacts/classifier",
          "uri": "s3://tempo/basic/pipeline", "platform": "tempo", "inputs": {"args":
          [{"ty": "numpy.ndarray", "name": "payload"}]}, "outputs": {"args": [{"ty": "numpy.ndarray",
          "name": null}, {"ty": "builtins.str", "name": null}]}, "description": "A pipeline
          to use either an sklearn or xgboost model for Iris classification"}, "protocol":
          "tempo.kfserving.protocol.KFServingV2Protocol", "runtime_options": {"runtime":
          "tempo.kfserving.KFServingKubernetesRuntime", "state_options": {"state_type":
          "LOCAL", "key_prefix": "", "host": "", "port": ""}, "insights_options": {"worker_endpoint":
          "", "batch_size": 1, "parallelism": 1, "retries": 3, "window_time": 0, "mode_type":
          "NONE", "in_asyncio": false}, "ingress_options": {"ingress": "tempo.ingress.istio.IstioIngress",
          "ssl": false, "verify_ssl": true}, "replicas": 1, "minReplicas": null, "maxReplicas":
          null, "authSecretName": null, "serviceAccountName": "kf-tempo", "add_svc_orchestrator":
          false, "namespace": "production"}}'
      labels:
        seldon.io/tempo: "true"
      name: classifier
      namespace: production
    spec:
      predictor:
        containers:
        - env:
          - name: STORAGE_URI
            value: s3://tempo/basic/pipeline
          - name: MLSERVER_HTTP_PORT
            value: "8080"
          - name: MLSERVER_GRPC_PORT
            value: "9000"
          - name: MLSERVER_MODEL_IMPLEMENTATION
            value: tempo.mlserver.InferenceRuntime
          - name: MLSERVER_MODEL_NAME
            value: classifier
          - name: MLSERVER_MODEL_URI
            value: /mnt/models
          - name: TEMPO_RUNTIME_OPTIONS
            value: '{"runtime": "tempo.kfserving.KFServingKubernetesRuntime", "state_options":
              {"state_type": "LOCAL", "key_prefix": "", "host": "", "port": ""}, "insights_options":
              {"worker_endpoint": "", "batch_size": 1, "parallelism": 1, "retries": 3,
              "window_time": 0, "mode_type": "NONE", "in_asyncio": false}, "ingress_options":
              {"ingress": "tempo.ingress.istio.IstioIngress", "ssl": false, "verify_ssl":
              true}, "replicas": 1, "minReplicas": null, "maxReplicas": null, "authSecretName":
              null, "serviceAccountName": "kf-tempo", "add_svc_orchestrator": false, "namespace":
              "production"}'
          image: seldonio/mlserver:0.3.2
          name: mlserver
          resources:
            limits:
              cpu: 1
              memory: 1Gi
            requests:
              cpu: 500m
              memory: 500Mi
        serviceAccountName: kf-tempo
    ---
    apiVersion: serving.kubeflow.org/v1beta1
    kind: InferenceService
    metadata:
      annotations:
        seldon.io/tempo-description: SKLearn Iris classification model
        seldon.io/tempo-model: '{"model_details": {"name": "test-iris-sklearn", "local_folder":
          "/home/clive/work/mlops/fork-tempo/docs/examples/kfserving/artifacts/sklearn",
          "uri": "s3://tempo/basic/sklearn", "platform": "sklearn", "inputs": {"args":
          [{"ty": "numpy.ndarray", "name": null}]}, "outputs": {"args": [{"ty": "numpy.ndarray",
          "name": null}]}, "description": "SKLearn Iris classification model"}, "protocol":
          "tempo.kfserving.protocol.KFServingV2Protocol", "runtime_options": {"runtime":
          "tempo.kfserving.KFServingKubernetesRuntime", "state_options": {"state_type":
          "LOCAL", "key_prefix": "", "host": "", "port": ""}, "insights_options": {"worker_endpoint":
          "", "batch_size": 1, "parallelism": 1, "retries": 3, "window_time": 0, "mode_type":
          "NONE", "in_asyncio": false}, "ingress_options": {"ingress": "tempo.ingress.istio.IstioIngress",
          "ssl": false, "verify_ssl": true}, "replicas": 1, "minReplicas": null, "maxReplicas":
          null, "authSecretName": null, "serviceAccountName": "kf-tempo", "add_svc_orchestrator":
          false, "namespace": "production"}}'
      labels:
        seldon.io/tempo: "true"
      name: test-iris-sklearn
      namespace: production
    spec:
      predictor:
        serviceAccountName: kf-tempo
        sklearn:
          protocolVersion: v2
          storageUri: s3://tempo/basic/sklearn
    ---
    apiVersion: serving.kubeflow.org/v1beta1
    kind: InferenceService
    metadata:
      annotations:
        seldon.io/tempo-description: XGBoost Iris classification model
        seldon.io/tempo-model: '{"model_details": {"name": "test-iris-xgboost", "local_folder":
          "/home/clive/work/mlops/fork-tempo/docs/examples/kfserving/artifacts/xgboost",
          "uri": "s3://tempo/basic/xgboost", "platform": "xgboost", "inputs": {"args":
          [{"ty": "numpy.ndarray", "name": null}]}, "outputs": {"args": [{"ty": "numpy.ndarray",
          "name": null}]}, "description": "XGBoost Iris classification model"}, "protocol":
          "tempo.kfserving.protocol.KFServingV2Protocol", "runtime_options": {"runtime":
          "tempo.kfserving.KFServingKubernetesRuntime", "state_options": {"state_type":
          "LOCAL", "key_prefix": "", "host": "", "port": ""}, "insights_options": {"worker_endpoint":
          "", "batch_size": 1, "parallelism": 1, "retries": 3, "window_time": 0, "mode_type":
          "NONE", "in_asyncio": false}, "ingress_options": {"ingress": "tempo.ingress.istio.IstioIngress",
          "ssl": false, "verify_ssl": true}, "replicas": 1, "minReplicas": null, "maxReplicas":
          null, "authSecretName": null, "serviceAccountName": "kf-tempo", "add_svc_orchestrator":
          false, "namespace": "production"}}'
      labels:
        seldon.io/tempo: "true"
      name: test-iris-xgboost
      namespace: production
    spec:
      predictor:
        serviceAccountName: kf-tempo
        xgboost:
          protocolVersion: v2
          storageUri: s3://tempo/basic/xgboost



```python

```
