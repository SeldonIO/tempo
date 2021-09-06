# MLflow end-to-end example

In this example we are going to build a model using `mlflow`, pack and deploy locally using `tempo` (in docker and local kubernetes cluster).

We are are going to use follow the MNIST pytorch example from `mlflow`, check this [link](https://github.com/mlflow/mlflow/tree/master/examples/pytorch/MNIST) for more information.



In this example we will:
  * [Train MNIST Model using mlflow and pytorch](#Train-model)
  * [Create tempo artifacts](#Save-model-environment)
  * [Deploy Locally to Docker](#Deploy-to-Docker)
  * [Deploy Locally to Kubernetes](#Deploy-to-Kubernetes)

## Prerequisites

This notebooks needs to be run in the `tempo-examples` conda environment defined below. Create from project root folder:

```bash
conda env create --name tempo-examples --file conda/tempo-examples.yaml
```

## Train model

We train MNIST model below:

### Install prerequisites


```python
!pip install mlflow torchvision>=0.9.1 torch==1.9.0 pytorch-lightning==1.4.0
```


```python
%cd /tmp
```


```python
!git clone https://github.com/mlflow/mlflow.git
```

### Train model using `mlflow`


```python
%cd mlflow/examples/pytorch/MNIST
```


```python
!mlflow run . --no-conda
```


```python
!tree -L 1 mlruns/0
```

### Choose test image


```python
from torchvision import datasets

mnist_test = datasets.MNIST('/tmp/data', train=False, download=True)
mnist_test = list(mnist_test)[0]
img, category = mnist_test
display(img)
print(category)
```

### Tranform test image to numpy


```python
import numpy as np
img_np = np.asarray(img).reshape((1, 28*28)).astype(np.float32)
```

## Save model environment


```python
import os
ARTIFACTS_FOLDER = os.path.join(
    os.getcwd(),
    "mlruns/0",
    os.listdir("mlruns/0")[0],
    "artifacts",
    "model"
)
print(ARTIFACTS_FOLDER)
```

### Define `tempo` model


```python
from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model

mlflow_tag = "mlflow"

pytorch_mnist_model = Model(
    name="test-pytorch-mnist",
    platform=ModelFramework.MLFlow,
    local_folder=ARTIFACTS_FOLDER,
    # if we deploy to kube, this defines where the model artifacts are stored
    uri="s3://tempo/basic/mnist",
    description="A pytorch MNIST model",
)


```

### Save model (environment) using `tempo`

Tempo hides many details required to save the model environment for `mlserver`:
- Add required runtime dependencies
- Create a conda pack `environment.tar.gz`


```python
from tempo.serve.loader import save
save(pytorch_mnist_model)
```

## Deploy to Docker


```python
from tempo import deploy_local
local_deployed_model = deploy_local(pytorch_mnist_model)
```


```python
local_prediction = local_deployed_model.predict(img_np)
print(np.nonzero(local_prediction.flatten() == 0))
```


```python
local_deployed_model.undeploy()
```

## Deploy to Kubernetes

### Prerequisites
 
Create a Kind Kubernetes cluster with Minio and Seldon Core installed using Ansible as described [here](https://tempo.readthedocs.io/en/latest/overview/quickstart.html#kubernetes-cluster-with-seldon-core).


```python
%cd -0
```


```python
!kubectl apply -f k8s/rbac -n seldon
```

### Upload artifacts to minio


```python
from tempo.examples.minio import create_minio_rclone
import os
create_minio_rclone(os.getcwd()+"/rclone.conf")
```


```python
from tempo.serve.loader import upload
upload(pytorch_mnist_model)
```


```python
from tempo.serve.metadata import SeldonCoreOptions
runtime_options = SeldonCoreOptions(**{
        "remote_options": {
            "namespace": "seldon",
            "authSecretName": "minio-secret"
        }
    })
```

### Deploy to `kind`


```python
from tempo import deploy_remote
remote_deployed_model = deploy_remote(pytorch_mnist_model, options=runtime_options)
```


```python
remote_prediction = remote_deployed_model.predict(img_np)
print(np.nonzero(remote_prediction.flatten() == 0))
```


```python
remote_deployed_model.undeploy()
```


```python

```
