```python
import numpy as np

from tempo.serve.metadata import ModelFramework, KubernetesOptions
from tempo.serve.model import Model
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.utils import pipeline, predictmethod
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.serve.utils import pipeline
```


```python
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("test")
```


```python
!kubectl create ns production
```


```python
!kubectl apply -f ../tempo/tests/testdata/tempo-pipeline-rbac.yaml -n production
```


```python
!helm upgrade --install redis bitnami/redis -n production \
    --set usePassword=false \
    --set master.service.type=LoadBalancer
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

k8s_options = KubernetesOptions(namespace="production")
k8s_runtime = SeldonKubernetesRuntime(k8s_options=k8s_options)

sklearn_model = Model(
        name="test-iris-sklearn",
        runtime=k8s_runtime,
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/mab/sklearn",
        local_folder=os.getcwd()+"/artifacts/mab/sklearn")

xgboost_model = Model(
        name="test-iris-xgboost",
        runtime=k8s_runtime,
        platform=ModelFramework.XGBoost,
        uri="gs://seldon-models/mab/xgboost",
        local_folder=os.getcwd()+"/artifacts/mab/xgboost/")
```


```python
sklearn_model.upload()
xgboost_model.upload()
```


```python
k8s_runtime_v2 = SeldonKubernetesRuntime(k8s_options=k8s_options, protocol=KFServingV2Protocol())

@pipeline(name="mab-pipeline",
          runtime=k8s_runtime_v2,
          uri="gs://seldon-models/mab/route",
          local_folder=os.getcwd()+"/artifacts/mab/route/",
          models=[sklearn_model, xgboost_model])
class MABRouter(object):

    def _init(self):
        self.n_branches = 2
        self.beta_params = [1 for _ in range(self.n_branches * 2)]
        
        import logging
        log = logging.getLogger(__name__)
        log.setLevel(10)
        self._log = log
        
        host = "redis-master"
        import os
        if os.environ.get("SELDON_LOCAL_ENVIRONMENT"):
            host = "localhost"
            
        self._log.info(f"Setting up redis with host {host}")
            
        import numpy as np
        self._np = np
        
        import redis
        self._rc = redis.Redis(host=host, port=6379)
        self._key = "seldon_deployment_predictor_model_1"
        
        if not self._rc.exists(self._key):
            models_beta_params = [1 for _ in range(self.n_branches * 2)]
            self._rc.lpush(self._key, *models_beta_params)
            
    @predictmethod
    def route(self, payload: np.ndarray) -> np.ndarray:

        if not hasattr(self, "_is_init") or not self._is_init:
            self._init()
            self._is_init = True
        
        models_beta_params = [float(i) for i in self._rc.lrange(self._key, 0, -1)]
        branch_values = [np.random.beta(a, b) for a, b in zip(*[iter(models_beta_params)] * 2)]
        selected_branch = np.argmax(branch_values)
        self._log.info(f"routing to branch: {selected_branch}")
        
        if selected_branch:
            return sklearn_model(payload)
        else:
            return xgboost_model(payload)
```


```python
%env SELDON_LOCAL_ENVIRONMENT=LOCAL

mab_router = MABRouter()
```


```python
mab_router.route(payload=X_rest[0:1])
```


```python
%%writefile artifacts/mab/route/conda.yaml
name: tempo
channels:
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - ca-certificates=2021.1.19=h06a4308_0
  - certifi=2020.12.5=py37h06a4308_0
  - ld_impl_linux-64=2.33.1=h53a641e_7
  - libedit=3.1.20191231=h14c3975_1
  - libffi=3.3=he6710b0_2
  - libgcc-ng=9.1.0=hdf63c60_0
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - ncurses=6.2=he6710b0_1
  - openssl=1.1.1j=h27cfd23_0
  - pip=21.0.1=py37h06a4308_0
  - python=3.7.9=h7579374_0
  - readline=8.1=h27cfd23_0
  - setuptools=52.0.0=py37h06a4308_0
  - sqlite=3.33.0=h62c20be_0
  - tk=8.6.10=hbc83047_0
  - wheel=0.36.2=pyhd3eb1b0_0
  - xz=5.2.5=h7b6447c_0
  - zlib=1.2.11=h7b6447c_3
  - pip:
    - redis==3.5.3
    - websocket-client==0.58.0
    - mlops-tempo==0.1.0.dev4
    - mlserver==0.3.1.dev5
    - mlserver-tempo==0.3.1.dev5
```


```python
# Currently needed as "save" doesn't fully work after sending a request
mab_router = MABRouter()
```


```python
mab_router.save(save_env=True)
```


```python
mab_router.upload()
```


```python
mab_router.deploy()
```


```python
mab_router.wait_ready()
```


```python
mab_router.remote(payload=X_rest[0:1])
```


```python
@pipeline(name="mab-feedback",
          runtime=k8s_runtime_v2,
          local_folder=os.getcwd()+"/artifacts/mab/feedback/",
          conda_env="tempo",
          uri="gs://seldon-models/custom")
class MABFeedback(object):

    def _init(self):
        self.n_branches = 2
        self.beta_params = [1 for _ in range(self.n_branches * 2)]
        
        import logging
        log = logging.getLogger(__name__)
        log.setLevel(10)
        self._log = log
        
        host = "redis-master"
        import os
        if os.environ.get("SELDON_LOCAL_ENVIRONMENT"):
            host = "localhost"
        
        self._log.info(f"Setting up redis with host {host}")
            
        import numpy as np
        self._np = np
        
        import redis
        self._rc = redis.Redis(host=host, port=6379)
        self._key = "seldon_deployment_predictor_model_1"
        
        if not self._rc.exists(self._key):
            models_beta_params = [1 for _ in range(self.n_branches * 2)]
            self._log.info(f"Creating new key in redis with vals: {models_beta_params}")
            self._rc.lpush(self._key, *models_beta_params)
        else:
            self._log.info("Redis key already exists")

    @predictmethod
    def feedback(self, payload: np.ndarray, parameters: dict) -> np.ndarray:

        if not hasattr(self, "_is_init") or not self._is_init:
            self._init()
            self._is_init = True
            
        self._log.info(f"Feedback method with truth {payload} and parameters {parameters}")
                
        reward = parameters["reward"]
        routing = parameters["routing"]

        self._log.info(f"Sending feedback with route {routing} reward {reward}")
        
        # Currently only support 1 feedback at a time
        n_predictions = 1
        n_success = int(reward * n_predictions)
        n_failures = n_predictions - n_success
    
        self._log.info(f"n_success: {n_success}, n_failures: {n_failures}")

        # Non atomic, race condition op
        self._log.info(f"LINDEX key {self._key} on index {routing*2}")
        success_val = float(self._rc.lindex(self._key, int(routing*2)))
        self._rc.lset(self._key, int(routing*2), str(success_val + n_success))
        fail_val = float(self._rc.lindex(self._key, int(routing*2 + 1)))
        self._rc.lset(self._key, int(routing*2 + 1), str(fail_val + n_failures))
        
        return np.array([n_success, n_failures])
        
```


```python
%env SELDON_LOCAL_ENVIRONMENT=LOCAL

mab_feedback = MABFeedback()
```


```python
X_rest[0:1]
```


```python
mab_feedback.feedback(payload=X_rest[0:1], parameters={ "reward": 1, "routing": 0} )
```


```python
%env SELDON_LOCAL_ENVIRONMENT=LOCAL

mab_feedback = MABFeedback()
```


```python
mab_feedback.save(save_env=False)
```


```python
mab_feedback.pipeline.upload()
```


```python
mab_feedback.deploy()
```


```python
mab_feedback.wait_ready()
```


```python
mab_feedback.remote(payload=X_rest[0:1], parameters={"reward":0.0,"routing":0} )
```


```python

```
