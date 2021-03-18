# Explanations Demo with Income Classifier and Alibi

## Conda env create

We create a conda environment for the runtime of our explainer from the `./artifacts/income_explainer/conda.yaml`
**This only needs to be done once**.


```python
!conda env create --name tempo-explainer-example --file ./artifacts/income_explainer/conda.yaml
```

## Setup and download Data


```python
from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.utils import pipeline, predictmethod
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.serve.metadata import ModelFramework, KubernetesOptions
from alibi.utils.wrappers import ArgmaxTransformer
from typing import Any

import numpy as np
import os 
import pprint
import dill
import json

EXPLAINER_FOLDER = os.getcwd()+"/artifacts/income_explainer"
MODEL_FOLDER = os.getcwd()+"/artifacts/income_model"
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
k8s_options = KubernetesOptions(namespace="production")
k8s_runtime = SeldonKubernetesRuntime(k8s_options=k8s_options)

sklearn_model = Model(
        name="income-sklearn",
        runtime=SeldonDockerRuntime(),
        platform=ModelFramework.SKLearn,
        local_folder=MODEL_FOLDER,
        uri="gs://seldon-models/test/income/model"
)

```


```python


@pipeline(name="income-explainer",
          runtime=SeldonDockerRuntime(protocol=KFServingV2Protocol()),
          uri="gs://seldon-models/test/income/explainer",
          local_folder=EXPLAINER_FOLDER,
          conda_env="tempo-explainer-example",
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

### Deploy model to Docker and test


```python
sklearn_model.deploy()
sklearn_model.wait_ready()
```


```python
sklearn_model(X_test[0:1])
```

### Create explainer and test against model


```python
p = ExplainerPipeline() 
```


```python
p.save(save_env=True)
```


```python
r = json.loads(p.explain(X_test[0:1], {"threshold":0.95}))
print(r["data"]["anchor"])
```

### Deploy explainer to docker


```python
p.deploy()
p.wait_ready()
```


```python
r = json.loads(p.remote(payload=X_test[0:1], parameters={"threshold":0.99}))
print(r["data"]["anchor"])
```


```python
p.undeploy()
```

### Deploy to production on Kubernetes


```python
k8s_options = KubernetesOptions(namespace="production")
k8s_runtime = SeldonKubernetesRuntime(k8s_options=k8s_options)
k8s_runtime_v2 = SeldonKubernetesRuntime(k8s_options=k8s_options, protocol=KFServingV2Protocol())

sklearn_model.set_runtime(k8s_runtime)
p.set_runtime(k8s_runtime_v2)
```


```python
p.save(save_env=False)
```

Upload artifacts. **This step may take some time**


```python
sklearn_model.upload()
p.upload()
```


```python
p.deploy()
p.wait_ready()
```


```python
r = json.loads(p.remote(payload=X_test[0:1], parameters={"threshold":0.95}))
print(r["data"]["anchor"])
```


```python
p.undeploy()
```

## Prepare for Gitops


```python
yaml = p.to_k8s_yaml()
print (eval(pprint.pformat(yaml)))
```


```python

```
