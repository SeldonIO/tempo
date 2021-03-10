# Deploying pipeline with custom environment

This notebook will walk you through an end-to-end example deploying a Tempo pipeline, running on its own Conda environment.

## Defining pipeline

The first step will be to define our custom pipeline.
This pipeline will access 2 models, stored remotely. 


```python
import numpy as np

from tempo.serve.metadata import ModelFramework, KubernetesOptions
from tempo.serve.model import Model
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.utils import pipeline, predictmethod
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.serve.utils import pipeline

k8s_options = KubernetesOptions(namespace="production")
k8s_runtime = SeldonKubernetesRuntime(k8s_options=k8s_options)

sklearn_model = Model(
        name="test-iris-sklearn",
        runtime=k8s_runtime,
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris"
)

xgboost_model = Model(
        name="test-iris-xgboost",
        runtime=k8s_runtime,
        platform=ModelFramework.XGBoost,
        uri="gs://seldon-models/xgboost/iris"
)

k8s_runtime_v2 = SeldonKubernetesRuntime(k8s_options=k8s_options, protocol=KFServingV2Protocol())

@pipeline(name="classifier",
          runtime=k8s_runtime_v2,
          uri="gs://seldon-models/test/custom",
          local_folder="./pipeline",
          models=[sklearn_model, xgboost_model])
def classifier(payload: np.ndarray) -> np.ndarray:
    res1 = sklearn_model(payload)

    if res1[0][0] > 0.7:
        return res1
    else:
        return xgboost_model(payload)
```

## Deploying pipeline

The next step, will be to deploy our pipeline to Kubernetes.
We will divide this process into 3 sub-steps:

1. Save our artifacts and environment
2. Upload to remote storage
3. Deploy resources

### Saving artifacts


```python
classifier.save()
```

### Uploading artifacts


```python
classifier.upload()
```

### Deploying pipeline

#### Setting up RBAC


```python
!kubectl apply -f ../tempo/tests/testdata/tempo-pipeline-rbac.yaml -n production
```


```python
classifier.deploy()
classifier.wait_ready()
```

## Sending requests

Lastly, we can now send requests to our deployed pipeline.
For this, we will leverage the `remote()` method, which will interact without our deployed pipeline (as opposed to executing our pipeline's code locally).


```python
classifier.remote(payload=np.array([[1, 2, 3, 4]]))
```


```python

```
