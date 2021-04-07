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
    - mlserver==0.3.1.dev7
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
    - mlserver==0.3.1.dev7
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


```python
docker_runtime = SeldonDockerRuntime()
```


```python
# docker_runtime.undeploy(model_sklearn)
# docker_runtime.undeploy(model_xgboost)
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
print(model_sklearn(payload=p1))
print(model_sklearn(payload=p2))
```


```python
print(model_sklearn.remote(payload=p1))
print(model_sklearn.remote(payload=p2))
```


```python
print(model_xgboost(payload=p1))
print(model_xgboost(payload=p2))
```


```python
print(model_xgboost.remote(payload=p1))
print(model_xgboost.remote(payload=p2))
```

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
def classifier(payload: np.ndarray) -> Tuple[dict, str]:
    res1 = classifier.models.sklearn(payload=payload)
    if res1["prediction"][0][0] > 0.5:
        return res1,"sklearn prediction"
    else:
        return classifier.models.xgboost(payload=payload), "xgboost prediction"
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
    - mlserver==0.3.1.dev5
```


```python
save(classifier, save_env=True)
```


```python
docker_runtime.deploy(classifier)
```


```python
classifier(payload=p1)
```


```python
classifier.remote(payload=p1)
```


```python
docker_runtime.undeploy(classifier)
```
