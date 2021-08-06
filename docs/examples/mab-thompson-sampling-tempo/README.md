# Multi-Armed Bandit with State


```python
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("test")
```


```python
!kaggle datasets download -d uciml/default-of-credit-card-clients-dataset
!unzip -o default-of-credit-card-clients-dataset.zip
```


```python
!mkdir -p artifacts/mab/
!mkdir -p artifacts/mab/route/
!mkdir -p artifacts/mab/feedback/
```


```python
import pandas as pd
data = pd.read_csv('UCI_Credit_Card.csv')
```


```python
target = 'default.payment.next.month'
```


```python
import numpy as np
from sklearn.model_selection import train_test_split

OBSERVED_DATA = 15000
TRAIN_1 = 10000
TEST_1 = 5000

REST_DATA = 15000

RUN_DATA = 5000
ROUTE_DATA = 10000

# get features and target
X = data.loc[:, data.columns!=target].values
y = data[target].values

# observed/unobserved split
X_obs, X_rest, y_obs, y_rest = train_test_split(X, y, random_state=1, test_size=REST_DATA)

# observed split into train1/test1
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_obs, y_obs, random_state=1, test_size=TEST_1)

# unobserved split into run/route
X_run, X_route, y_run, y_route = train_test_split(X_rest, y_rest, random_state=1, test_size=ROUTE_DATA)

# observed+run split into train2/test2
X_rest = np.vstack((X_run, X_route))
y_rest = np.hstack((y_run, y_route))

X_train2 = np.vstack((X_train1, X_test1))
X_test2 = X_run

y_train2 = np.hstack((y_train1, y_test1))
y_test2 = y_run
```


```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train1, y_train1)
```


```python
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=1)
xgb.fit(X_train2, y_train2)
```


```python
!mkdir -p artifacts/mab/sklearn/
!mkdir -p artifacts/mab/xgboost/
```


```python
import joblib
joblib.dump(rf, 'artifacts/mab/sklearn/model.joblib')
```


```python
xgb.save_model('artifacts/mab/xgboost/model.bst')
```


```python
import os
from tempo.serve.model import Model
from tempo.serve.metadata import ModelFramework

sklearn_tempo = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/mab/sklearn",
        local_folder=os.getcwd()+"/artifacts/mab/sklearn")

xgboost_tempo = Model(
        name="test-iris-xgboost",
        platform=ModelFramework.XGBoost,
        uri="gs://seldon-models/mab/xgboost",
        local_folder=os.getcwd()+"/artifacts/mab/xgboost/")
```


```python
from tempo import deploy_local
remote_sklearn = deploy_local(sklearn_tempo)
remote_xgboost = deploy_local(xgboost_tempo)
```


```python
remote_sklearn.predict(X_test2[0:1])
```


```python
remote_xgboost.predict(X_test2[0:1])
```


```python
from tempo.docker.utils import deploy_redis

deploy_redis()
```


```python
import logging

import numpy as np
from tempo.serve.utils import pipeline, predictmethod
from tempo.serve.metadata import InsightRequestModes, SeldonCoreOptions, StateTypes
from tempo.serve.constants import DefaultRedisLocalHost, DefaultRedisPort, DefaultRedisK8sHost
from tempo.serve.pipeline import PipelineModels

from tempo.magic import t

runtime_options = SeldonCoreOptions(**{
    "local_options": {
        "state_options": {
            "state_type": StateTypes.REDIS,
            "host": DefaultRedisLocalHost,
            "port": DefaultRedisPort,
        }
    },
    "remote_options": {
        "state_options": {
            "state_type": StateTypes.REDIS,
            "host": DefaultRedisK8sHost,
            "port": DefaultRedisPort,
        }
    }
})

@pipeline(name="mab-router",
          runtime_options=runtime_options.local_options,
          uri="s3://tempo/mab/route",
          local_folder=os.getcwd()+"/artifacts/mab/router/",
          models=PipelineModels(sklearn=sklearn_tempo, xgboost=xgboost_tempo))
class MABRouter(object):

    def __init__(self):
        self.n_branches = 2
        self.beta_params = [1 for _ in range(self.n_branches * 2)]
                            
        logging.info(f"Setting up MAB routing pipeline")
            
        self._key = "beta_params"
            
    @predictmethod
    def predict(self, payload: np.ndarray) -> np.ndarray:
        
        if not t.state.exists(self._key):
            models_beta_params = [1 for _ in range(self.n_branches * 2)]
            t.state.internal_state.lpush(self._key, *models_beta_params)
        
        models_beta_params = [float(i) for i in t.state.internal_state.lrange(self._key, 0, -1)]
        branch_values = [np.random.beta(a, b) for a, b in zip(*[iter(models_beta_params)] * 2)]
        selected_branch = np.argmax(branch_values)
        logging.info(f"routing to branch: {selected_branch}")
        
        if selected_branch:
            return self.models.xgboost(payload)
        else:
            return self.models.sklearn(payload)
```


```python
mab_router = MABRouter()
```


```python
for i in range(10):
    print(mab_router.predict(X_test2[0:1]))
```


```python
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
```


```python
import os

TEMPO_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
```


```python
%%writetemplate artifacts/mab/router/conda.yaml
name: tempo-insights
channels:
  - defaults
dependencies:
  - pip=21.0.1
  - python=3.7.9
  - pip:
    - mlops-tempo @ file://{TEMPO_DIR}
    - mlserver==0.3.1.dev7
```


```python
from tempo.serve.loader import save
save(mab_router, save_env=True)
```


```python
from tempo import deploy_local
from tempo.serve.constants import DefaultRedisDockerHost

runtime_options.local_options.state_options.host = DefaultRedisDockerHost

remote_mab_router = deploy_local(mab_router, runtime_options)
```


```python
for i in range(10):
    print(remote_mab_router.predict(X_test2[0:1]))
```


```python
@pipeline(name="mab-feedback",
          runtime_options=runtime_options.local_options,
          uri="s3://tempo/mab/feedback",
          local_folder=os.getcwd()+"/artifacts/mab/feedback/")
class MABFeedback(object):

    def __init__(self):
        self._key = "beta_params"

    @predictmethod
    def predict(self, payload: np.ndarray, parameters: dict) -> np.ndarray:
            
        logging.info(f"Feedback method with truth {payload} and parameters {parameters}")
                
        reward = parameters["reward"]
        routing = parameters["routing"]

        logging.info(f"Sending feedback with route {routing} reward {reward}")
        
        # Currently only support 1 feedback at a time
        n_predictions = 1
        n_success = int(reward * n_predictions)
        n_failures = n_predictions - n_success
    
        logging.info(f"n_success: {n_success}, n_failures: {n_failures}")

        # Non atomic, race condition op
        logging.info(f"LINDEX key {self._key} on index {routing*2}")
        success_val = float(t.state.internal_state.lindex(self._key, int(routing*2)))
        t.state.internal_state.lset(self._key, int(routing*2), str(success_val + n_success))
        fail_val = float(t.state.internal_state.lindex(self._key, int(routing*2 + 1)))
        t.state.internal_state.lset(self._key, int(routing*2 + 1), str(fail_val + n_failures))
        
        return np.array([n_success, n_failures])
        
```


```python
mab_feedback = MABFeedback()
```

### Deploy Feedback Pipeline


```python
%%writetemplate artifacts/mab/feedback/conda.yaml
name: tempo-insights
channels:
  - defaults
dependencies:
  - pip=21.0.1
  - python=3.7.9
  - pip:
    - mlops-tempo @ file://{TEMPO_DIR}
    - mlserver==0.3.1.dev7
```


```python
save(mab_feedback, save_env=True)
```


```python
remote_mab_feedback = deploy_local(mab_feedback, runtime_options)
```

## Send feedback showing that route sklearn model performs better


```python
for i in range(10):
    print(remote_mab_feedback.predict(payload=X_rest[0:1], parameters={ "reward": 1, "routing": 0}))
```

## See now most requests being sent to sklearn model


```python
for i in range(10):
    print(remote_mab_router.predict(X_test2[0:1]))
```

### Now send 20 positive requests showing xgboost performing better


```python
for i in range(20):
    print(remote_mab_feedback.predict(payload=X_rest[0:1], parameters={ "reward": 1, "routing": 1}))
```

### We should now see the xgboost model receiving most requests


```python
for i in range(10):
    print(remote_mab_router.predict(X_test2[0:1]))
```

### Clean up Docker


```python
remote_mab_router.undeploy()
remote_mab_feedback.undeploy()
```


```python
from tempo.docker.utils import undeploy_redis
undeploy_redis()
```

## Deploy to Kubernetes


```python
!kubectl create ns production
```


```python
!kubectl apply -f k8s/rbac -n default
```


```python
from tempo.examples.minio import create_minio_rclone
import os
create_minio_rclone(os.getcwd()+"/rclone.conf")
```


```python
from tempo.serve.loader import upload
upload(mab_router)
upload(mab_feedback)
```


```python
from tempo.k8s.utils import deploy_redis

deploy_redis()
```


```python
from tempo import deploy_remote
k8s_mab_router = deploy_remote(mab_router, options=runtime_options)
k8s_mab_feedback = deploy_remote(mab_feedback, options=runtime_options)
```


```python
k8s_mab_router.predict(payload=X_rest[0:1])
```


```python
k8s_mab_feedback.predict(payload=X_rest[0:1], parameters={"reward":0.0,"routing":0} )
```


```python
k8s_mab_router.undeploy()
k8s_mab_feedback.undeploy()
```


```python

```
