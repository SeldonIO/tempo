# Outlier Detection with CIFAR10 using Alibi-Detect

This e



## Prerequisites

This notebooks needs to be run in the `tempo-examples` conda environment defined below. Create with:

```bash
conda env create --name tempo-examples --file tempo-examples.yaml
```


```python
!cat ../../../conda/tempo-examples.yaml
```


```python
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
```


```python
import cloudpickle
import tensorflow as tf
import matplotlib.pyplot as plt

from tempo.serve.metadata import ModelFramework, RuntimeOptions
from tempo.serve.model import Model
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.kfserving.protocol import KFServingV2Protocol, KFServingV1Protocol
from tempo.serve.utils import pipeline, predictmethod, model
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.kfserving.k8s import KFServingKubernetesRuntime
from tempo.serve.metadata import ModelFramework, KubernetesOptions
from tempo.serve.loader import save, upload, download
from alibi.utils.wrappers import ArgmaxTransformer
from typing import Any

import numpy as np
import os 
import pprint
import json

OUTLIER_FOLDER = os.getcwd()+"/artifacts/cifar10_outlier"
MODEL_FOLDER = os.getcwd()+"/artifacts/cifar10_model"
SVC_FOLDER = os.getcwd()+"/artifacts/svc"

import logging
logging.basicConfig(level=logging.INFO)
```


```python
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Layer, Reshape, InputLayer
from tqdm import tqdm

from alibi_detect.models.losses import elbo
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.fetching import fetch_detector
from alibi_detect.utils.perturbation import apply_mask
from alibi_detect.utils.saving import load_detector
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
```


```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # pylint: disable=arguments-differ,method-hidden
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

```


```python
def show_image(X):
    plt.imshow(X.reshape(32, 32, 3))
    plt.axis('off')
    plt.show()
```


```python
train, test = tf.keras.datasets.cifar10.load_data()
X_train, y_train = train
X_test, y_test = test

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
```


```python
%%writefile rclone.conf
[gcs]
type = google cloud storage
anonymous = true
```


```python
load_pretrained = True
if load_pretrained:  # load pre-trained detector
    !rclone copy gs://seldon-models/tempo/cifar10/outlier/cifar10 ./artifacts/cifar10_outlier/cifar10
else:  # define model, initialize, train and save outlier detector

    # define encoder and decoder networks
    latent_dim = 1024
    encoder_net = tf.keras.Sequential(
      [
          InputLayer(input_shape=(32, 32, 3)),
          Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu)
      ]
    )

    decoder_net = tf.keras.Sequential(
      [
          InputLayer(input_shape=(latent_dim,)),
          Dense(4*4*128),
          Reshape(target_shape=(4, 4, 128)),
          Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
      ]
    )

    # initialize outlier detector
    od = OutlierVAE(
        threshold=.015,  # threshold for outlier score
        encoder_net=encoder_net,  # can also pass VAE model instead
        decoder_net=decoder_net,  # of separate encoder and decoder
        latent_dim=latent_dim
    )

    # train
    od.fit(X_train, epochs=50, verbose=False)

    # save the trained outlier detector
    save_detector(od, filepath)
```

## Create Tempo Artifacts


```python
runtimeOptions=RuntimeOptions(  
                              k8s_options=KubernetesOptions( 
                                        namespace="production",
                                        authSecretName="minio-secret")
                              )


cifar10_model = Model(
        name="resnet32",
        protocol=KFServingV1Protocol(),
        runtime_options=runtimeOptions,
        platform=ModelFramework.Tensorflow,
        uri="gs://seldon-models/tfserving/cifar10/resnet32",
        local_folder=MODEL_FOLDER,
    )

@model(
        name="outlier",
        platform=ModelFramework.TempoPipeline,
        protocol=KFServingV2Protocol(),
        runtime_options=runtimeOptions,
        uri="s3://tempo/outlier/cifar10/outlier",
        local_folder=OUTLIER_FOLDER,
    )
class OutlierModel(object):
    
    def __init__(self):
        self.loaded = False
        
    def load(self):
        if "MLSERVER_MODELS_DIR" in os.environ:
            models_folder = "/mnt/models"
        else:
            models_folder = OUTLIER_FOLDER
        self.od = load_detector(models_folder+"/cifar10")
        self.loaded = True
        
    def unload(self):
        self.od = None
        self.loaded = False
        
    @predictmethod
    def outlier(self, payload: np.ndarray) -> dict:
        if not self.loaded:
            self.load()
        print("Outlier called")
        od_preds = self.od.predict(payload,
                      outlier_type='instance',    # use 'feature' or 'instance' level
                      return_feature_score=True,  # scores used to determine outliers
                      return_instance_score=True)
        
        return json.loads(json.dumps(od_preds, cls=NumpyEncoder))
    
outlier = OutlierModel()

@pipeline(
        name="cifar10-service",
        protocol=KFServingV2Protocol(),
        runtime_options=runtimeOptions,
        uri="s3://tempo/outlier/cifar10/svc",
        local_folder=SVC_FOLDER,
        models=[outlier, cifar10_model]
    )
class Cifar10(object):
        
    @predictmethod
    def predict(self, payload: np.ndarray) -> np.ndarray:
        r = outlier(payload=payload)
        if r["data"]["is_outlier"][0]:
            return np.array([])
        else:
            return cifar10_model(payload)

        
svc = Cifar10()
```


```python
idx = 1
X = X_train[idx:idx+1]
np.random.seed(0) 
X_mask, mask = apply_mask(X.reshape(1, 32, 32, 3),
                                  mask_size=(10,10),
                                  n_masks=1,
                                  channels=[0,1,2],
                                  mask_type='normal',
                                  noise_distr=(0,1),
                                  clip_rng=(0,1))
```

## Saving Artifacts


```python
import sys
import os
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
TEMPO_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
```


```python
%%writetemplate artifacts/cifar10_outlier/conda.yaml
name: tempo
channels:
  - defaults
dependencies:
  - python={PYTHON_VERSION}
  - pip:
    - alibi-detect
    - dill
    - opencv-python-headless
    - mlops-tempo @ file://{TEMPO_DIR}
    - mlserver==0.3.1.dev7
```

FIXME: remove alibi-detect from pip in below when isolated


```python
%%writetemplate artifacts/svc/conda.yaml
name: tempo
channels:
  - defaults
dependencies:
  - python={PYTHON_VERSION}
  - pip:
    - alibi-detect
    - dill
    - opencv-python-headless
    - mlops-tempo @ file://{TEMPO_DIR}
    - mlserver==0.3.1.dev7
```


```python
outlier = OutlierModel()
save(outlier, save_env=True)
```


```python
svc = Cifar10()
save(svc, save_env=True)
```

## Test on docker with local pipelines


```python
download(cifar10_model)
```

## Test Svc on docker


```python
docker_runtime = SeldonDockerRuntime()
docker_runtime.deploy(svc)
docker_runtime.wait_ready(svc)
```


```python
show_image(X_test[0:1])
svc.remote(payload=X_test[0:1])
```


```python
show_image(X_mask)
svc.remote(payload=X_mask)
```

## Deploy to Kubernetes

To create a Kubernetes cluster in Kind with all the required install use the ansible playbook defined  below.
Create with

```bash
ansible-playbook ansible/playbooks/default.yaml
```


```python
!cat ../../../ansible/playbooks/default.yaml
```


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
[gcs]
type = google cloud storage
anonymous = true

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
upload(outlier)
upload(svc)
```

## Deploy to Kubernetes with Seldon


```python
k8s_runtime = SeldonKubernetesRuntime()
k8s_runtime.deploy(svc)
k8s_runtime.wait_ready(svc)
```


```python
from tempo.utils import tempo_settings
tempo_settings.remote_kubernetes(True)
```


```python
show_image(X_test[0:1])
svc.remote(payload=X_test[0:1])
```


```python
show_image(X_mask)
svc.remote(payload=X_mask)
```


```python
k8s_runtime.undeploy(svc)
```


```python

```
