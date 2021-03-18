# Outlier Detection with CIFAR10 using Alibi-Detect



## Conda env create

We create a conda environment for the runtime of our explainer from the `./artifacts/income_explainer/conda.yaml`
**This only needs to be done once**.


```python
!conda env create --name tempo-outlier-example --file ./artifacts/cifar10_outlier/conda.yaml
```


```python
import cloudpickle
import tensorflow as tf
import matplotlib.pyplot as plt

from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.kfserving.protocol import KFServingV2Protocol, KFServingV1Protocol
from tempo.serve.utils import pipeline, predictmethod, model
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.serve.metadata import ModelFramework, KubernetesOptions
from alibi.utils.wrappers import ArgmaxTransformer
from typing import Any

import numpy as np
import os 
import pprint
import json

OUTLIER_FOLDER = os.getcwd()+"/artifacts/cifar10_outlier"
MODEL_FOLDER = os.getcwd()+"/artifacts/cifar10_model"
SVC_FOLDER = os.getcwd()+"/artifacts/svc"
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

## Create Tempo Artifacts


```python
cifar10_model = Model(
        name="resnet32",
        runtime=SeldonDockerRuntime(protocol=KFServingV1Protocol()),
        platform=ModelFramework.Tensorflow,
        uri="gs://seldon-models/tfserving/cifar10/resnet32",
        local_folder=MODEL_FOLDER,
    )

@model(
        name="outlier",
        platform=ModelFramework.TempoPipeline,
        runtime=SeldonDockerRuntime(protocol=KFServingV2Protocol()),
        uri="gs://seldon-models/tempo/cifar10/outlier",
        conda_env="tempo-outlier-example",
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
        runtime=SeldonDockerRuntime(protocol=KFServingV2Protocol()),
        uri="gs://seldon-models/tempo/cifar10/svc",
        conda_env="tempo-outlier-example",
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

## Test on docker with local pipelines


```python
cifar10_model.download()
```


```python
cifar10_model.deploy()
cifar10_model.wait_ready()
```


```python
idx = 1
X = X_test[idx].reshape(1, 32, 32, 3)
plt.imshow(X.reshape(32, 32, 3))
plt.axis('off')
plt.show()
print("class:",class_names[y_test[idx][0]])
print("prediction:",class_names[cifar10_model(X_test[idx:idx+1])[0].argmax()])
```


```python
show_image(X_mask)
r = outlier(payload=X_mask)
print("Is outlier:",r["data"]["is_outlier"][0] == 1)
```


```python
svc.predict(X_test[idx:idx+1])
```


```python
svc.predict(X_mask)
```

## Test on Outlier on docker 


```python
outlier.unload()
outlier.save(save_env=True)
```


```python
outlier.deploy()
```


```python
show_image(X_mask)
r = outlier.remote(payload=X_mask)
print("Is outlier:",r["data"]["is_outlier"][0] == 1)
```


```python
show_image(X_test[0:1])
r = outlier.remote(payload=X_test[0:1])
print("Is outlier:",r["data"]["is_outlier"][0] == 1)
```

## Test Svc on docker


```python
outlier.unload()
svc.save(save_env=True)
```


```python
svc.deploy()
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


```python
k8s_options = KubernetesOptions(namespace="production")
k8s_v1_runtime = SeldonKubernetesRuntime(k8s_options=k8s_options, protocol=KFServingV1Protocol())
k8s_v2_runtime = SeldonKubernetesRuntime(k8s_options=k8s_options, protocol=KFServingV2Protocol())

cifar10_model.set_runtime(k8s_v1_runtime)
outlier.set_runtime(k8s_v2_runtime)
svc.set_runtime(k8s_v2_runtime)
```


```python
outlier.save(save_env=True)
outlier.upload()
```


```python
#svc.save(save_env=False)
svc.upload()
```


```python
svc.deploy()
svc.wait_ready()
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
svc.undeploy()
```


```python

```
