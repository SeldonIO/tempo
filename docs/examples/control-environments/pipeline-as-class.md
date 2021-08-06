# Control the runtime environment


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
import os
```


```python
XGBOOST_FOLDER = f"{os.getcwd()}/artifacts/xgboost"
SKLEARN_FOLDER = f"{os.getcwd()}/artifacts/sklearn"
```


```python
!mkdir -p {XGBOOST_FOLDER}
!mkdir -p {SKLEARN_FOLDER}
```

## Train Models


```python
import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import joblib

iris = datasets.load_iris()
X, y = iris.data, iris.target

logreg = LogisticRegression(C=1e5)
logreg.fit(X, y)

with open(f"{SKLEARN_FOLDER}/model.joblib","wb") as f:
    joblib.dump(logreg, f)
```


```python
import xgboost
clf = xgboost.XGBClassifier()
clf.fit(X, y)
clf.save_model(f"{XGBOOST_FOLDER}/model.bst")
```

    [14:37:34] WARNING: ../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.


    /home/rskolasinski/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/xgboost/sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


## Write models environments


```python
import sys
import os
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
SKLEARN_VERSION = sklearn.__version__
XGBOOST_VERSION = xgboost.__version__
TEMPO_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
```


```python
%%writetemplate $SKLEARN_FOLDER/conda.yaml
name: tempo-sklearn
channels:
  - defaults
dependencies:
  - python={PYTHON_VERSION}
  - pip:
    - mlops-tempo @ file://{TEMPO_DIR}
    - scikit-learn=={SKLEARN_VERSION}
    - mlserver==0.3.1.dev5
    - mlserver-tempo==0.3.1.dev5
```


```python
%%writetemplate $XGBOOST_FOLDER/conda.yaml
name: tempo-xgboost
channels:
  - defaults
dependencies:
  - python={PYTHON_VERSION}
  - pip:
    - mlops-tempo @ file://{TEMPO_DIR}
    - xgboost=={XGBOOST_VERSION}
    - mlserver==0.3.1.dev5
    - mlserver-tempo==0.3.1.dev5
```

## Define Model Servers


```python
from tempo.serve.metadata import ModelFramework, KubernetesOptions

from tempo.kfserving.protocol import KFServingV2Protocol

from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.seldon.docker import SeldonDockerRuntime

import tempo.serve.utils as tempo_utils
from tempo.serve.loader import save
```


```python
import numpy as np
import socket
```


```python
import logging
logging.basicConfig(level=logging.INFO)
```


```python
from typing import Dict, Any

import joblib
import socket
from typing import Tuple
import xgboost as xgb

@tempo_utils.model(
    name="sklearn-classifier",
    platform=ModelFramework.TempoPipeline,
    uri="s3://tempo/control-environments/iris",
    local_folder=SKLEARN_FOLDER,
)
class IrisClassifier:
    def __init__(self):
        self.ready = False
      
    def load(self):
        try:
            self.model = joblib.load("/mnt/models/model.joblib")        
            self.ready = True
        except FileNotFoundError:
            self.model = joblib.load(f"{SKLEARN_FOLDER}/model.joblib")        
            self.ready = True

    @tempo_utils.predictmethod
    def predict(self, payload: np.ndarray) -> dict:
        if not self.ready:
            self.load()
        prediction = self.model.predict_proba(payload)
        return {"prediction": prediction.tolist(), "meta": {"hostname": socket.gethostname()}}
    
    
@tempo_utils.model(
    name="xgboost-classifier",
    platform=ModelFramework.TempoPipeline,
    uri="s3://tempo/control-environments/xgboost",
    local_folder=XGBOOST_FOLDER,
)
class XGBoostClassifier:
    def __init__(self):
        self.ready = False

    def load(self):
        try:
            self.model = xgb.Booster(model_file="/mnt/models/model.bst")
            self.ready = True
        except (FileNotFoundError, Exception):
            self.model = xgb.Booster(model_file=f"{XGBOOST_FOLDER}/model.bst")        
            self.ready = True            

    @tempo_utils.predictmethod
    def predict(self, payload: np.ndarray) -> dict:
        if not self.ready:
            self.load()
        prediction = self.model.predict(xgb.DMatrix(payload))
        return {"prediction": prediction.tolist(), "meta": {"hostname": socket.gethostname()}}
```


```python
model_sklearn = IrisClassifier()
model_xgboost = XGBoostClassifier()
```


```python
save(model_sklearn, save_env=True)
save(model_xgboost, save_env=True)
```

    INFO:tempo:Saving environment
    INFO:tempo:Saving tempo model to /home/rskolasinski/work/tempo/docs/examples/control-environments/artifacts/sklearn/model.pickle
    INFO:tempo:Using found conda.yaml
    INFO:tempo:Creating conda env with: conda env create --name tempo-7c4f55a2-0760-46fb-9a31-6c586c08fef1 --file /tmp/tmpbb26jkmn.yml
    INFO:tempo:packing conda environment from tempo-7c4f55a2-0760-46fb-9a31-6c586c08fef1


    Collecting packages...
    Packing environment at '/home/rskolasinski/miniconda3/envs/tempo-7c4f55a2-0760-46fb-9a31-6c586c08fef1' to '/home/rskolasinski/work/tempo/docs/examples/control-environments/artifacts/sklearn/environment.tar.gz'
    [########################################] | 100% Completed | 15.5s


    INFO:tempo:Removing conda env with: conda remove --name tempo-7c4f55a2-0760-46fb-9a31-6c586c08fef1 --all --yes
    INFO:tempo:Saving environment
    INFO:tempo:Saving tempo model to /home/rskolasinski/work/tempo/docs/examples/control-environments/artifacts/xgboost/model.pickle
    INFO:tempo:Using found conda.yaml
    INFO:tempo:Creating conda env with: conda env create --name tempo-f43d828c-54af-4182-b0ef-4632585f88c9 --file /tmp/tmpkr72i37t.yml
    INFO:tempo:packing conda environment from tempo-f43d828c-54af-4182-b0ef-4632585f88c9


    Collecting packages...
    Packing environment at '/home/rskolasinski/miniconda3/envs/tempo-f43d828c-54af-4182-b0ef-4632585f88c9' to '/home/rskolasinski/work/tempo/docs/examples/control-environments/artifacts/xgboost/environment.tar.gz'
    [########################################] | 100% Completed | 23.1s


    INFO:tempo:Removing conda env with: conda remove --name tempo-f43d828c-54af-4182-b0ef-4632585f88c9 --all --yes



```python
docker_runtime = SeldonDockerRuntime()
```


```python
# model_sklearn.undeploy()
# model_xgboost.undeploy()
```


```python
docker_runtime.deploy(model_sklearn)
docker_runtime.deploy(model_xgboost)
```


```python
import numpy as np

p1 = np.array([[1, 2, 3, 4]])
p2 = np.array([[5.964,4.006,2.081,1.031]])
```


```python
print(model_sklearn.remote(payload=p1))
print(model_sklearn.remote(payload=p2))
```

    {'prediction': [[9.49810079285076e-34, 2.267015334079471e-19, 1.0]], 'meta': {'hostname': '7a192a0249d1'}}
    {'prediction': [[0.9999999998972331, 1.0276696730328812e-10, 1.633959045505507e-30]], 'meta': {'hostname': '7a192a0249d1'}}



```python
print(model_xgboost.remote(payload=p1))
print(model_xgboost.remote(payload=p2))
```

    {'prediction': [[0.00847206823527813, 0.03168793022632599, 0.9598399996757507]], 'meta': {'hostname': '09a8c18c9bee'}}
    {'prediction': [[0.9732961654663086, 0.024121448397636414, 0.002582334913313389]], 'meta': {'hostname': '09a8c18c9bee'}}


## Define Pipeline


```python
PIPELINE_FOLDER = f"{os.getcwd()}/artifacts/classifier"
!mkdir -p {PIPELINE_FOLDER}
```


```python
from tempo.serve.pipeline import PipelineModels
from typing import Tuple


@tempo_utils.pipeline(
    name="classifier",
    uri="s3://tempo/control-environments/classifier",
    local_folder=PIPELINE_FOLDER,
    models=PipelineModels(sklearn=model_sklearn, xgboost=model_xgboost)
)
class Classifier:
    @tempo_utils.predictmethod
    def classifier(self, payload: np.ndarray) -> Tuple[dict, str]:
        res1 = self.models.sklearn(payload=payload)

        if res1["prediction"][0][0] > 0.5:
            return res1,"sklearn prediction"
        else:
            return self.models.xgboost(payload=payload), "xgboost prediction"
        
        
classifier = Classifier()        
```


```python
%%writetemplate $PIPELINE_FOLDER/conda.yaml
name: tempo
channels:
  - defaults
dependencies:
  - python={PYTHON_VERSION}
  - pip:
    - mlops-tempo @ file://{TEMPO_DIR}
    - joblib
    - mlserver==0.3.1.dev7
```


```python
docker_runtime = SeldonDockerRuntime()
save(classifier, save_env=True)
```

    INFO:tempo:Saving environment
    INFO:tempo:Saving tempo model to /home/rskolasinski/work/tempo/docs/examples/control-environments/artifacts/classifier/model.pickle
    INFO:tempo:Using found conda.yaml
    INFO:tempo:Creating conda env with: conda env create --name tempo-36628da6-b0fa-4a31-adf9-4dfb8a3592d8 --file /tmp/tmpaq44de1z.yml
    INFO:tempo:packing conda environment from tempo-36628da6-b0fa-4a31-adf9-4dfb8a3592d8


    Collecting packages...
    Packing environment at '/home/rskolasinski/miniconda3/envs/tempo-36628da6-b0fa-4a31-adf9-4dfb8a3592d8' to '/home/rskolasinski/work/tempo/docs/examples/control-environments/artifacts/classifier/environment.tar.gz'
    [########################################] | 100% Completed | 12.2s


    INFO:tempo:Removing conda env with: conda remove --name tempo-36628da6-b0fa-4a31-adf9-4dfb8a3592d8 --all --yes



```python
docker_runtime.deploy(classifier)
```


```python
classifier(payload=p1)
```




    ({'prediction': [[0.00847206823527813,
        0.03168793022632599,
        0.9598399996757507]],
      'meta': {'hostname': 'machine42'}},
     'xgboost prediction')




```python
classifier.remote(payload=p1)
```




    {'output0': {'prediction': [[0.00847206823527813,
        0.03168793022632599,
        0.9598399996757507]],
      'meta': {'hostname': '09a8c18c9bee'}},
     'output1': 'xgboost prediction'}




```python
docker_runtime.undeploy(classifier)
```

    INFO:tempo:Undeploying classifier
    INFO:tempo:Undeploying sklearn-classifier
    INFO:tempo:Undeploying xgboost-classifier

