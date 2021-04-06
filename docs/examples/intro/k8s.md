# Introction to Tempo on Kubernetes

This notebook will walk you through an end-to-end example deploying a Tempo pipeline deployed to Kubernetes.

## Prerequisites

This notebooks needs to be run in the `tempo-examples` conda environment defined below. Create from project root folder:

```bash
conda env create --name tempo-examples --file conda/tempo-examples.yaml
```

## Architecture

We will show two Iris dataset prediction models combined with service orchestration that shows some arbitrary python code to control predictions from the two models.

![architecture](architecture.png)


```python
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
```

## Train Iris Models

We will train:

  * A sklearn logistic regression model
  * A xgboost model


```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import joblib
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target
logreg = LogisticRegression(C=1e5)
logreg.fit(X, y)
logreg.predict_proba(X[0:1])
with open("./artifacts/sklearn/model.joblib","wb") as f:
    joblib.dump(logreg, f)
```


```python
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X,y)
clf.save_model("./artifacts/xgboost/model.bst")
```

## Defining pipeline

The first step will be to define our custom pipeline.
This pipeline will access 2 models, stored remotely. 


```python
import os
import numpy as np
from typing import Tuple
from tempo.serve.metadata import ModelFramework, KubernetesOptions, RuntimeOptions
from tempo.serve.model import Model
from tempo.seldon.protocol import SeldonProtocol
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.utils import pipeline, predictmethod
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.serve.utils import pipeline
from tempo.serve.loader import save, upload

import logging
logging.basicConfig(level=logging.INFO)

SKLEARN_FOLDER = os.getcwd()+"/artifacts/sklearn"
XGBOOST_FOLDER = os.getcwd()+"/artifacts/xgboost"
PIPELINE_ARTIFACTS_FOLDER = os.getcwd()+"/artifacts/classifier"

runtimeOptions=RuntimeOptions(  
                              k8s_options=KubernetesOptions( 
                                        namespace="production",
                                        authSecretName="minio-secret")
                              )

sklearn_model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        protocol=SeldonProtocol(),
        runtime_options=runtimeOptions,
        local_folder=SKLEARN_FOLDER,
        uri="s3://tempo/basic/sklearn"
)

xgboost_model = Model(
        name="test-iris-xgboost",
        platform=ModelFramework.XGBoost,
        protocol=SeldonProtocol(),
        runtime_options=runtimeOptions,    
        local_folder=XGBOOST_FOLDER,
        uri="s3://tempo/basic/xgboost"
)

@pipeline(name="classifier",
          uri="s3://tempo/basic/pipeline",
          local_folder=PIPELINE_ARTIFACTS_FOLDER,
          runtime_options=runtimeOptions,
          models=[sklearn_model, xgboost_model])
def classifier(payload: np.ndarray) -> Tuple[np.ndarray,str]:
    res1 = sklearn_model(payload)

    if res1[0][0] > 0.5:
        return res1,"sklearn prediction"
    else:
        return xgboost_model(payload),"xgboost prediction"
```

### Saving artifacts

We provide a conda yaml in out `local_folder` which tempo will use as the runtime environment to save. If this file was not there it would save the current conda environment. One can also provide a named conda environment with `conda_env` in the decorator.


```python
import sys
import os
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
TEMPO_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
```


```python
%%writetemplate artifacts/classifier/conda.yaml
name: tempo
channels:
  - defaults
dependencies:
  - python={PYTHON_VERSION}
  - pip:
    - mlops-tempo @ file://{TEMPO_DIR}
    - mlserver==0.3.1.dev7
```


```python
save(classifier, save_env=True)
```

## Deploying pipeline to K8s

The next step, will be to deploy our pipeline to Kubernetes.
We will divide this process into 3 sub-steps:

1. Save our artifacts and environment
2. Upload to remote storage
3. Deploy resources

### Setup Namespace with Minio Secret


```python
!kubectl create namespace production
```


```python
!kubectl apply -f ../../../k8s/tempo-pipeline-rbac.yaml -n production
```


```python
%%writefile minio-secret.yaml

apiVersion: v1
kind: Secret
metadata:
  name: minio-secret
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: minioadmin
  AWS_SECRET_ACCESS_KEY: minioadmin
  AWS_ENDPOINT_URL: http://minio.minio-system.svc.cluster.local:9000
  USE_SSL: "false"
```


```python
!kubectl apply -f minio-secret.yaml -n production
```

### Uploading artifacts


```python
MINIO_IP=!kubectl get svc minio -n minio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
MINIO_IP=MINIO_IP[0]
```


```python
%%writetemplate rclone.conf
[s3]
type = s3
provider = minio
env_auth = false
access_key_id = minioadmin
secret_access_key = minioadmin
endpoint = http://{MINIO_IP}:9000
```


```python
import os
from tempo.conf import settings
settings.rclone_cfg = os.getcwd() + "/rclone.conf"
```


```python
upload(sklearn_model)
upload(xgboost_model)
upload(classifier)
```

### Deploy


```python
k8s_runtime = SeldonKubernetesRuntime()
k8s_runtime.deploy(classifier)
k8s_runtime.wait_ready(classifier)
```

### Sending requests

Lastly, we can now send requests to our deployed pipeline.
For this, we will leverage the `remote()` method, which will interact without our deployed pipeline (as opposed to executing our pipeline's code locally).


```python
from tempo.utils import tempo_settings
tempo_settings.remote_kubernetes(True)
```


```python
classifier(payload=np.array([[1, 2, 3, 4]]))
```


```python
classifier.remote(payload=np.array([[1, 2, 3, 4]]))
```


```python
classifier.remote(payload=np.array([[5.964,4.006,2.081,1.031]]))
```

### Undeploy pipeline


```python
k8s_runtime.undeploy(classifier)
```


```python

```
