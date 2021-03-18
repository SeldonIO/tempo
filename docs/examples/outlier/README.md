```python
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
import dill
import json

OUTLIER_FOLDER = os.getcwd()+"/artifacts/cifar10_outlier"
MODEL_FOLDER = os.getcwd()+"/artifacts/cifar10_model"
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
from alibi_detect.utils.saving import save_detector, load_detector
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

cifar10_outlier = Model(
        name="resnet32_outlier",
        runtime=SeldonDockerRuntime(protocol=KFServingV1Protocol()),
        platform=ModelFramework.Alibi-Detect,
        uri="gs://seldon-models/test/cifar10/outlier",
        local_folder=MODEL_FOLDER,
    )


@pipeline(
        name="outlier",
        runtime=SeldonDockerRuntime(protocol=KFServingV1Protocol()),
        uri="gs://seldon-models/test/cifar10/outlier",
        conda_env="tempo-outlier-example",
        local_folder=OUTLIER_FOLDER,
    )


class OutlierModel(object):
    
    def __init__(self):
        self.od = load_detector(OUTLIER_FOLDER+"/cifar10")
        
    @predictmethod
    def outlier(self, payload: np.ndarray) -> str:
        print("Outlier called")
        od_preds = self.od.predict(payload,
                      outlier_type='instance',    # use 'feature' or 'instance' level
                      return_feature_score=True,  # scores used to determine outliers
                      return_instance_score=True)
        
        return json.loads(json.dumps(od_preds, cls=NumpyEncoder))
    
outlier = OutlierModel()

@pipeline(
        name="cifar10-service",
        runtime=SeldonDockerRuntime(protocol=KFServingV1Protocol()),
        uri="gs://seldon-models/test/cifar10/svc",
        conda_env="tempo-minimal",
        local_folder=OUTLIER_FOLDER,
        models=[outlier,cifar10_model]
    )
class Cifar10(object):
    
    def __init__(self):
        self.od = load_detector(OUTLIER_FOLDER+"/cifar10")
        
    @predictmethod
    def svc(self, payload: np.ndarray) -> str:
        r = outlier(payload)
        if r["outlier"]:
            return "outlier"
        else:
            return cifar10_model(payload)
```


```python
o = OutlierModel()
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


```python
show_image(X_mask)
o.outlier(X_mask)
```

## Test on docker


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
cifar10_model.undeploy()
```

## Deploy to Kubernetes


```python
k8s_options = KubernetesOptions(namespace="production")
k8s_runtime = SeldonKubernetesRuntime(k8s_options=k8s_options, protocol=KFServingV1Protocol())

cifar10_model.set_runtime(k8s_runtime)
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
cifar10_model.undeploy()
```


```python

```
