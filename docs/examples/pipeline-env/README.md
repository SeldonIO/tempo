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

docker_runtime = SeldonDockerRuntime()

sklearn_model = Model(
        name="test-iris-sklearn",
        runtime=docker_runtime,
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris"
)

xgboost_model = Model(
        name="test-iris-xgboost",
        runtime=docker_runtime,
        platform=ModelFramework.XGBoost,
        uri="gs://seldon-models/xgboost/iris"
)

docker_runtime_v2 = SeldonDockerRuntime(protocol=KFServingV2Protocol())

@pipeline(name="classifier",
          runtime=docker_runtime_v2,
          uri="gs://seldon-models/test/custom",
          models=[sklearn_model, xgboost_model])
def classifier(payload: np.ndarray) -> np.ndarray:
    res1 = sklearn_model(payload)

    if res1[0][0] > 0.7:
        return res1
    else:
        return xgboost_model(payload)
```

## Deploying pipeline to Docker

The next step, will be to deploy our pipeline to Docker.
We will divide this process into 3 sub-steps:

1. Save our artifacts and environment
2. Download our model artifacts locally
3. Deploy resources

### Saving artifacts


```python
classifier.save()
```

### Downloading model artifacts

Since we are going to deploy our pipeline locally using Docker, we'll need to download the model artifacts locally.


```python
sklearn_model.download()
xgboost_model.download()
```

### Deploying pipeline


```python
classifier.deploy()
classifier.wait_ready()
```

### Sending requests

Lastly, we can now send requests to our deployed pipeline.
For this, we will leverage the `remote()` method, which will interact without our deployed pipeline (as opposed to executing our pipeline's code locally).


```python
classifier.remote(payload=np.array([[1, 2, 3, 4]]))
```

### Undeploy pipeline


```python
classifier.undeploy()
```

## Deploying pipeline to K8s

The next step, will be to deploy our pipeline to Kubernetes.
We will divide this process into 3 sub-steps:

1. Save our artifacts and environment
2. Upload to remote storage
3. Deploy resources

### Change runtime


```python
k8s_options = KubernetesOptions(namespace="production")
k8s_runtime = SeldonKubernetesRuntime(k8s_options=k8s_options)

sklearn_model.set_runtime(k8s_runtime)
xgboost_model.set_runtime(k8s_runtime)

k8s_runtime_v2 = SeldonKubernetesRuntime(k8s_options=k8s_options, protocol=KFServingV2Protocol())

classifier.set_runtime(k8s_runtime_v2)
```

### Saving artifacts


```python
classifier.save(save_env=False)
```

### Uploading artifacts


```python
classifier.upload()
```

### Setting up RBAC


```python
!kubectl apply -f ../../../tempo/tests/testdata/tempo-pipeline-rbac.yaml -n production
```

### Deploy


```python
classifier.deploy()
classifier.wait_ready()
```

### Sending requests

Lastly, we can now send requests to our deployed pipeline.
For this, we will leverage the `remote()` method, which will interact without our deployed pipeline (as opposed to executing our pipeline's code locally).


```python
classifier.remote(payload=np.array([[1, 2, 3, 4]]))
```

### Undeploy pipeline


```python
classifier.undeploy()
```
