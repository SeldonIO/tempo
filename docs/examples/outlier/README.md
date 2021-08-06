# Outlier Example

![architecture](architecture.png)

In this example we will:

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
    │   ├── [01;34mmodel[00m
    │   ├── [01;34moutlier[00m
    │   └── [01;34msvc[00m
    ├── [01;34mk8s[00m
    │   └── [01;34mrbac[00m
    ├── [01;34mREADME_files[00m
    ├── [01;34msrc[00m
    │   ├── constants.py
    │   ├── data.py
    │   ├── outlier.py
    │   ├── tempo.py
    │   └── utils.py
    └── [01;34mtests[00m
        └── test_tempo.py
    
    9 directories, 6 files


## Train Models

 * This section is where as a data scientist you do your work of training models and creating artfacts.
 * For this example we train sklearn and xgboost classification models for the iris dataset.


```python
import os
import logging
import numpy as np
import tempo

from tempo.utils import logger
from src.constants import ARTIFACTS_FOLDER

logger.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
```


```python
from src.data import Cifar10
data = Cifar10()
```

    (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)


Download pretrained Resnet32 Tensorflow model for CIFAR10


```python
!rclone --config ./rclone-gcs.conf copy gs://seldon-models/tfserving/cifar10/resnet32 ./artifacts/model
```

Download or train an outlier detector on CIFAR10 data


```python
load_pretrained = True
if load_pretrained:  # load pre-trained detector
    !rclone --config ./rclone-gcs.conf copy gs://seldon-models/tempo/cifar10/outlier/cifar10 ./artifacts/outlier/cifar10
else:
    from src.outlier import train_outlier_detector
    train_outlier_detector(data, ARTIFACTS_FOLDER)
```

## Create Tempo Artifacts



```python
from src.tempo import create_outlier_cls, create_model, create_svc_cls

cifar10_model = create_model()
OutlierModel = create_outlier_cls()
outlier = OutlierModel()
Cifar10Svc = create_svc_cls(outlier, cifar10_model)
svc = Cifar10Svc()
```

    Loading from /home/clive/work/mlops/fork-tempo/docs/examples/outlier/artifacts/outlier



```python
# %load src/tempo.py
import json
import os

import numpy as np
from alibi_detect.base import NumpyEncoder
from src.constants import ARTIFACTS_FOLDER, MODEL_FOLDER, OUTLIER_FOLDER

from tempo.kfserving.protocol import KFServingV1Protocol, KFServingV2Protocol
from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.serve.pipeline import PipelineModels
from tempo.serve.utils import model, pipeline, predictmethod


def create_outlier_cls():
    @model(
        name="outlier",
        platform=ModelFramework.Custom,
        protocol=KFServingV2Protocol(),
        uri="s3://tempo/outlier/cifar10/outlier",
        local_folder=os.path.join(ARTIFACTS_FOLDER, OUTLIER_FOLDER),
    )
    class OutlierModel(object):
        def __init__(self):
            from alibi_detect.utils.saving import load_detector

            model = self.get_tempo()
            models_folder = model.details.local_folder
            print(f"Loading from {models_folder}")
            self.od = load_detector(os.path.join(models_folder, "cifar10"))

        @predictmethod
        def outlier(self, payload: np.ndarray) -> dict:
            od_preds = self.od.predict(
                payload,
                outlier_type="instance",  # use 'feature' or 'instance' level
                return_feature_score=True,
                # scores used to determine outliers
                return_instance_score=True,
            )

            return json.loads(json.dumps(od_preds, cls=NumpyEncoder))

    return OutlierModel


def create_model():

    cifar10_model = Model(
        name="resnet32",
        protocol=KFServingV1Protocol(),
        platform=ModelFramework.Tensorflow,
        uri="gs://seldon-models/tfserving/cifar10/resnet32",
        local_folder=os.path.join(ARTIFACTS_FOLDER, MODEL_FOLDER),
    )

    return cifar10_model


def create_svc_cls(outlier, model):
    @pipeline(
        name="cifar10-service",
        protocol=KFServingV2Protocol(),
        uri="s3://tempo/outlier/cifar10/svc",
        local_folder=os.path.join(ARTIFACTS_FOLDER, "svc"),
        models=PipelineModels(outlier=outlier, cifar10=model),
    )
    class Cifar10Svc(object):
        @predictmethod
        def predict(self, payload: np.ndarray) -> np.ndarray:
            r = self.models.outlier(payload=payload)
            if r["data"]["is_outlier"][0]:
                return np.array([])
            else:
                return self.models.cifar10(payload)

    return Cifar10Svc

```

## Unit Tests

 * Here we run our unit tests to ensure the orchestration works before running on the actual models.


```python
# %load tests/test_tempo.py
import numpy as np
from src.tempo import create_model, create_outlier_cls, create_svc_cls


def test_svc_outlier():
    model = create_model()
    OutlierModel = create_outlier_cls()
    outlier = OutlierModel()
    Cifar10Svc = create_svc_cls(outlier, model)
    svc = Cifar10Svc()
    svc.models.outlier = lambda payload: {"data": {"is_outlier": [1]}}
    svc.models.cifar10 = lambda input: np.array([[0.2]])
    res = svc(np.array([1]))
    assert res.shape[0] == 0


def test_svc_inlier():
    model = create_model()
    OutlierModel = create_outlier_cls()
    outlier = OutlierModel()
    Cifar10Svc = create_svc_cls(outlier, model)
    svc = Cifar10Svc()
    svc.models.outlier = lambda payload: {"data": {"is_outlier": [0]}}
    svc.models.cifar10 = lambda input: np.array([[0.2]])
    res = svc(np.array([1]))
    assert res.shape[0] == 1

```


```python
!python -m pytest tests/
```

    [1m============================= test session starts ==============================[0m
    platform linux -- Python 3.7.9, pytest-6.2.0, py-1.10.0, pluggy-0.13.1
    rootdir: /home/clive/work/mlops/fork-tempo, configfile: setup.cfg
    plugins: cases-3.4.6, asyncio-0.14.0
    collected 2 items                                                              [0m[1m
    
    tests/test_tempo.py [32m.[0m[32m.[0m[33m                                                   [100%][0m
    
    [33m=============================== warnings summary ===============================[0m
    ../../../../../../anaconda3/envs/tempo-examples/lib/python3.7/site-packages/tensorflow/python/autograph/impl/api.py:22
      /home/clive/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/tensorflow/python/autograph/impl/api.py:22: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
        import imp
    
    ../../../../../../anaconda3/envs/tempo-examples/lib/python3.7/site-packages/packaging/version.py:130
      /home/clive/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/packaging/version.py:130: DeprecationWarning: Creating a LegacyVersion has been deprecated and will be removed in the next major release
        DeprecationWarning,
    
    -- Docs: https://docs.pytest.org/en/stable/warnings.html
    [33m======================== [32m2 passed[0m, [33m[1m2 warnings[0m[33m in 3.77s[0m[33m =========================[0m
    Unresolved object in checkpoint: (root).encoder.fc_mean.kernel
    Unresolved object in checkpoint: (root).encoder.fc_mean.bias
    Unresolved object in checkpoint: (root).encoder.fc_log_var.kernel
    Unresolved object in checkpoint: (root).encoder.fc_log_var.bias
    A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
    Unresolved object in checkpoint: (root).encoder.fc_mean.kernel
    Unresolved object in checkpoint: (root).encoder.fc_mean.bias
    Unresolved object in checkpoint: (root).encoder.fc_log_var.kernel
    Unresolved object in checkpoint: (root).encoder.fc_log_var.bias
    A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.


## Save Outlier and Svc Environments



```python
tempo.save(OutlierModel)
```

    Collecting packages...
    Packing environment at '/home/clive/anaconda3/envs/tempo-c08c4322-62be-4461-82bc-d69ae2432671' to '/home/clive/work/mlops/fork-tempo/docs/examples/outlier/artifacts/outlier/environment.tar.gz'
    [########################################] | 100% Completed |  1min  9.5s



```python
tempo.save(Cifar10Svc)
```

    Collecting packages...
    Packing environment at '/home/clive/anaconda3/envs/tempo-27f221b3-8635-4b7e-ace6-443f6d7e3b15' to '/home/clive/work/mlops/fork-tempo/docs/examples/outlier/artifacts/svc/environment.tar.gz'
    [########################################] | 100% Completed | 11.5s


## Test Locally on Docker

Here we test our models using production images but running locally on Docker. This allows us to ensure the final production deployed model will behave as expected when deployed.


```python
from tempo import deploy_local
remote_model = deploy_local(svc)
```


```python
from src.utils import show_image
show_image(data.X_test[0:1])
remote_model.predict(payload=data.X_test[0:1])
```


    
![png](README_files/README_22_0.png)
    





    array([[3.92254496e-09, 1.20455460e-11, 2.66011191e-09, 9.99992609e-01,
            2.52213306e-10, 5.40860242e-07, 6.75954425e-06, 4.75119076e-12,
            6.90874735e-09, 1.07275586e-11]])




```python
from src.utils import create_cifar10_outlier

outlier_img = create_cifar10_outlier(data)
show_image(outlier_img)
remote_model.predict(payload=outlier_img)
```


    
![png](README_files/README_23_0.png)
    





    array([], dtype=float64)




```python
remote_model.undeploy()
```

## Production Option 1 (Deploy to Kubernetes with Tempo)

 * Here we illustrate how to run the final models in "production" on Kubernetes by using Tempo to deploy
 
### Prerequisites
 
Create a Kind Kubernetes cluster with Minio and Seldon Core installed using Ansible as described [here](https://tempo.readthedocs.io/en/latest/overview/quickstart.html#kubernetes-cluster-with-seldon-core).


```python
!kubectl apply -f k8s/rbac -n production
```

    secret/minio-secret configured
    serviceaccount/tempo-pipeline unchanged
    role.rbac.authorization.k8s.io/tempo-pipeline unchanged
    rolebinding.rbac.authorization.k8s.io/tempo-pipeline-rolebinding unchanged



```python
from tempo.examples.minio import create_minio_rclone
import os

create_minio_rclone(os.getcwd()+"/rclone-minio.conf")
```


```python
tempo.upload(cifar10_model)
tempo.upload(outlier)
tempo.upload(svc)
```


```python
from tempo.serve.metadata import SeldonCoreOptions
runtime_options = SeldonCoreOptions(**{
        "remote_options": {
            "namespace": "production",
            "authSecretName": "minio-secret"
        }
    })
```


```python
from tempo import deploy_remote
remote_model = deploy_remote(svc, options=runtime_options)
```


```python
from src.utils import show_image

show_image(data.X_test[0:1])
remote_model.predict(payload=data.X_test[0:1])
```


    
![png](README_files/README_31_0.png)
    





    array([[3.92254496e-09, 1.20455460e-11, 2.66011191e-09, 9.99992609e-01,
            2.52213306e-10, 5.40860242e-07, 6.75954425e-06, 4.75119076e-12,
            6.90874735e-09, 1.07275586e-11]])




```python
from src.utils import create_cifar10_outlier

outlier_img = create_cifar10_outlier(data)
show_image(outlier_img)
remote_model.predict(payload=outlier_img)
```


    
![png](README_files/README_32_0.png)
    





    array([], dtype=float64)




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
            "namespace": "production",
            "authSecretName": "minio-secret"
        }
    })
yaml_str = manifest(svc, options=runtime_options)
with open(os.getcwd()+"/k8s/tempo.yaml","w") as f:
    f.write(yaml_str)
```


```python
!kustomize build k8s
```

    apiVersion: machinelearning.seldon.io/v1
    kind: SeldonDeployment
    metadata:
      annotations:
        seldon.io/tempo-description: ""
        seldon.io/tempo-model: '{"model_details": {"name": "cifar10-service", "local_folder":
          "/home/clive/work/mlops/fork-tempo/docs/examples/outlier/artifacts/svc", "uri":
          "s3://tempo/outlier/cifar10/svc", "platform": "tempo", "inputs": {"args": [{"ty":
          "numpy.ndarray", "name": "payload"}]}, "outputs": {"args": [{"ty": "numpy.ndarray",
          "name": null}]}, "description": ""}, "protocol": "tempo.kfserving.protocol.KFServingV2Protocol",
          "runtime_options": {"runtime": "tempo.seldon.SeldonKubernetesRuntime", "state_options":
          {"state_type": "LOCAL", "key_prefix": "", "host": "", "port": ""}, "insights_options":
          {"worker_endpoint": "", "batch_size": 1, "parallelism": 1, "retries": 3, "window_time":
          0, "mode_type": "NONE", "in_asyncio": false}, "ingress_options": {"ingress":
          "tempo.ingress.istio.IstioIngress", "ssl": false, "verify_ssl": true}, "replicas":
          1, "minReplicas": null, "maxReplicas": null, "authSecretName": "minio-secret",
          "serviceAccountName": null, "add_svc_orchestrator": false, "namespace": "production"}}'
      labels:
        seldon.io/tempo: "true"
      name: cifar10-service
      namespace: production
    spec:
      predictors:
      - annotations:
          seldon.io/no-engine: "true"
        componentSpecs:
        - spec:
            containers:
            - name: classifier
              resources:
                limits:
                  cpu: 1
                  memory: 1Gi
                requests:
                  cpu: 500m
                  memory: 500Mi
        graph:
          envSecretRefName: minio-secret
          implementation: TEMPO_SERVER
          modelUri: s3://tempo/outlier/cifar10/svc
          name: cifar10-service
          serviceAccountName: tempo-pipeline
          type: MODEL
        name: default
        replicas: 1
      protocol: kfserving
    ---
    apiVersion: machinelearning.seldon.io/v1
    kind: SeldonDeployment
    metadata:
      annotations:
        seldon.io/tempo-description: ""
        seldon.io/tempo-model: '{"model_details": {"name": "outlier", "local_folder":
          "/home/clive/work/mlops/fork-tempo/docs/examples/outlier/artifacts/outlier",
          "uri": "s3://tempo/outlier/cifar10/outlier", "platform": "custom", "inputs":
          {"args": [{"ty": "numpy.ndarray", "name": "payload"}]}, "outputs": {"args":
          [{"ty": "builtins.dict", "name": null}]}, "description": ""}, "protocol": "tempo.kfserving.protocol.KFServingV2Protocol",
          "runtime_options": {"runtime": "tempo.seldon.SeldonKubernetesRuntime", "state_options":
          {"state_type": "LOCAL", "key_prefix": "", "host": "", "port": ""}, "insights_options":
          {"worker_endpoint": "", "batch_size": 1, "parallelism": 1, "retries": 3, "window_time":
          0, "mode_type": "NONE", "in_asyncio": false}, "ingress_options": {"ingress":
          "tempo.ingress.istio.IstioIngress", "ssl": false, "verify_ssl": true}, "replicas":
          1, "minReplicas": null, "maxReplicas": null, "authSecretName": "minio-secret",
          "serviceAccountName": null, "add_svc_orchestrator": false, "namespace": "production"}}'
      labels:
        seldon.io/tempo: "true"
      name: outlier
      namespace: production
    spec:
      predictors:
      - annotations:
          seldon.io/no-engine: "true"
        componentSpecs:
        - spec:
            containers:
            - args: []
              env:
              - name: MLSERVER_HTTP_PORT
                value: "9000"
              - name: MLSERVER_GRPC_PORT
                value: "9500"
              - name: MLSERVER_MODEL_IMPLEMENTATION
                value: tempo.mlserver.InferenceRuntime
              - name: MLSERVER_MODEL_NAME
                value: outlier
              - name: MLSERVER_MODEL_URI
                value: /mnt/models
              - name: TEMPO_RUNTIME_OPTIONS
                value: '{"runtime": "tempo.seldon.SeldonKubernetesRuntime", "state_options":
                  {"state_type": "LOCAL", "key_prefix": "", "host": "", "port": ""}, "insights_options":
                  {"worker_endpoint": "", "batch_size": 1, "parallelism": 1, "retries":
                  3, "window_time": 0, "mode_type": "NONE", "in_asyncio": true}, "ingress_options":
                  {"ingress": "tempo.ingress.istio.IstioIngress", "ssl": false, "verify_ssl":
                  true}, "replicas": 1, "minReplicas": null, "maxReplicas": null, "authSecretName":
                  "minio-secret", "serviceAccountName": null, "add_svc_orchestrator":
                  false, "namespace": "production"}'
              image: seldonio/mlserver:0.3.2
              name: outlier
        graph:
          envSecretRefName: minio-secret
          implementation: TEMPO_SERVER
          modelUri: s3://tempo/outlier/cifar10/outlier
          name: outlier
          serviceAccountName: tempo-pipeline
          type: MODEL
        name: default
        replicas: 1
      protocol: kfserving
    ---
    apiVersion: machinelearning.seldon.io/v1
    kind: SeldonDeployment
    metadata:
      annotations:
        seldon.io/tempo-description: ""
        seldon.io/tempo-model: '{"model_details": {"name": "resnet32", "local_folder":
          "/home/clive/work/mlops/fork-tempo/docs/examples/outlier/artifacts/model", "uri":
          "gs://seldon-models/tfserving/cifar10/resnet32", "platform": "tensorflow", "inputs":
          {"args": [{"ty": "numpy.ndarray", "name": null}]}, "outputs": {"args": [{"ty":
          "numpy.ndarray", "name": null}]}, "description": ""}, "protocol": "tempo.kfserving.protocol.KFServingV1Protocol",
          "runtime_options": {"runtime": "tempo.seldon.SeldonKubernetesRuntime", "state_options":
          {"state_type": "LOCAL", "key_prefix": "", "host": "", "port": ""}, "insights_options":
          {"worker_endpoint": "", "batch_size": 1, "parallelism": 1, "retries": 3, "window_time":
          0, "mode_type": "NONE", "in_asyncio": false}, "ingress_options": {"ingress":
          "tempo.ingress.istio.IstioIngress", "ssl": false, "verify_ssl": true}, "replicas":
          1, "minReplicas": null, "maxReplicas": null, "authSecretName": "minio-secret",
          "serviceAccountName": null, "add_svc_orchestrator": false, "namespace": "production"}}'
      labels:
        seldon.io/tempo: "true"
      name: resnet32
      namespace: production
    spec:
      predictors:
      - annotations:
          seldon.io/no-engine: "true"
        graph:
          envSecretRefName: minio-secret
          implementation: TENSORFLOW_SERVER
          modelUri: gs://seldon-models/tfserving/cifar10/resnet32
          name: resnet32
          type: MODEL
        name: default
        replicas: 1
      protocol: tensorflow



```python

```
