# Explanations Demo with Income Classifier and Alibi

## Conda env create

We create a conda environment for the runtime of our explainer from the `./artifacts/income_explainer/conda.yaml`
**This only needs to be done once**.


```python
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
```

## Setup and download Data


```python
from tempo.serve.metadata import ModelFramework, KubernetesOptions, RuntimeOptions
from tempo.serve.model import Model
from tempo.seldon.protocol import SeldonProtocol
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.utils import pipeline, predictmethod
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.serve.metadata import ModelFramework, KubernetesOptions
from alibi.utils.wrappers import ArgmaxTransformer
from tempo.serve.loader import save, upload
from typing import Any

import numpy as np
import os 
import pprint
import dill
import json

EXPLAINER_FOLDER = os.getcwd()+"/artifacts/income_explainer"
MODEL_FOLDER = os.getcwd()+"/artifacts/income_model"

import logging
logging.basicConfig(level=logging.INFO)
```


```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from alibi.explainers import AnchorTabular
from alibi.datasets import fetch_adult

adult = fetch_adult()
data = adult.data
target = adult.target
feature_names = adult.feature_names
category_map = adult.category_map
```


```python
np.random.seed(0)
data_perm = np.random.permutation(np.c_[data, target])
data = data_perm[:,:-1]
target = data_perm[:,-1]
idx = 30000
X_train,Y_train = data[:idx,:], target[:idx]
X_test, Y_test = data[idx+1:,:], target[idx+1:]
```

## Build SKLearn Model and Alibi Anchors Tabular Explainer


```python
ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])
categorical_features = list(category_map.keys())
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                               ('cat', categorical_transformer, categorical_features)])
clf = RandomForestClassifier(n_estimators=50)
model=Pipeline(steps=[("preprocess",preprocessor),("model",clf)])
model.fit(X_train,Y_train)


print('Train accuracy: ', accuracy_score(Y_train, model.predict(X_train)))
print('Test accuracy: ', accuracy_score(Y_test, model.predict(X_test)))
```


```python
from alibi.explainers import AnchorTabular
predict_fn = lambda x: model.predict(x)
explainer = AnchorTabular(predict_fn, feature_names, categorical_names=category_map, seed=1)
explainer.fit(X_train, disc_perc=[25, 50, 75])
```


```python
explanation = explainer.explain(X_test[0], threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)
```


```python
from joblib import dump
dump(model, MODEL_FOLDER+"/model.joblib") 
with open(EXPLAINER_FOLDER+"/explainer.dill", 'wb') as f:
    dill.dump(explainer,f)
```

## Create Tempo Artifacts


```python

runtimeOptions=RuntimeOptions(  
                              k8s_options=KubernetesOptions( 
                                        namespace="production",
                                        authSecretName="minio-secret")
                              )


sklearn_model = Model(
        name="income-sklearn",
        platform=ModelFramework.SKLearn,
        protocol=SeldonProtocol(),
        runtime_options=runtimeOptions,
        local_folder=MODEL_FOLDER,
        uri="gs://seldon-models/test/income/model"
)


@pipeline(name="income-explainer",
          uri="s3://tempo/explainer/pipeline",
          local_folder=EXPLAINER_FOLDER,
          runtime_options=runtimeOptions,
          models=[sklearn_model])
class ExplainerPipeline(object):

    def __init__(self):
        if "MLSERVER_MODELS_DIR" in os.environ:
            models_folder = ""
        else:
            models_folder = EXPLAINER_FOLDER
        with open(models_folder+"/explainer.dill", "rb") as f:
            self.explainer = dill.load(f)
        self.ran_init = True
        
    def update_predict_fn(self, x):
        if np.argmax(sklearn_model(x).shape) == 0:
            self.explainer.predictor = sklearn_model
            self.explainer.samplers[0].predictor = sklearn_model
        else:
            self.explainer.predictor = ArgmaxTransformer(sklearn_model)
            self.explainer.samplers[0].predictor = ArgmaxTransformer(sklearn_model)

    @predictmethod
    def explain(self, payload: np.ndarray, parameters: dict) -> str:
        print("Explain called with ", parameters)
        if not self.ran_init:
            print("Loading explainer")
            self.__init__()
        self.update_predict_fn(payload)
        explanation = self.explainer.explain(payload, **parameters)
        return explanation.to_json()
```

### Saving Artifacts


```python
import sys
import os
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
TEMPO_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
```


```python
%%writetemplate artifacts/income_explainer/conda.yaml
name: tempo
channels:
  - defaults
dependencies:
  - python={PYTHON_VERSION}
  - pip:
    - alibi
    - dill
    - mlops-tempo @ file://{TEMPO_DIR}
    - mlserver==0.3.1.dev7
```


```python
explainer = ExplainerPipeline()
save(explainer, save_env=True)
```

### Deploy explainer to docker


```python
docker_runtime = SeldonDockerRuntime()
docker_runtime.deploy(explainer)
docker_runtime.wait_ready(explainer)
```


```python
r = json.loads(explainer(payload=X_test[0:1], parameters={"threshold":0.99}))
print(r["data"]["anchor"])
```


```python
r = json.loads(explainer.remote(payload=X_test[0:1], parameters={"threshold":0.99}))
print(r["data"]["anchor"])
```


```python
docker_runtime.undeploy(explainer)
```

## Deploy to production on Kubernetes

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
upload(explainer)
```

### Deploy


```python
k8s_runtime = SeldonKubernetesRuntime()
k8s_runtime.deploy(explainer)
k8s_runtime.wait_ready(explainer)
```


```python
from tempo.utils import tempo_settings
tempo_settings.remote_kubernetes(True)
```


```python
r = json.loads(explainer.remote(payload=X_test[0:1], parameters={"threshold":0.95}))
print(r["data"]["anchor"])
```


```python
k8s_runtime.undeploy(explainer)
```

## Prepare for Gitops


```python
yaml = k8s_runtime.to_k8s_yaml(explainer)
print (eval(pprint.pformat(yaml)))
```


```python

```
