```python
from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.serve.utils import pipeline, predictmethod
from tempo.seldon.k8s import SeldonKubernetesRuntime

import numpy as np
import os 
import pprint
```


```python
sklearn_model = Model(
        name="test-iris-sklearn",
        runtime=SeldonDockerRuntime(),
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris",
        local_folder=os.getcwd()+"/sklearn")

xgboost_model = Model(
        name="test-iris-xgboost",
        runtime=SeldonDockerRuntime(),
        platform=ModelFramework.XGBoost,
        uri="gs://seldon-models/xgboost/iris",
        local_folder=os.getcwd()+"/xgboost")

@pipeline(name="mypipeline",
          runtime=SeldonDockerRuntime(protocol=KFServingV2Protocol()),
          uri="gs://seldon-models/custom",
          models=[sklearn_model, xgboost_model])
class MyPipeline(object):

    def __init__(self):
        self.counter = 0

    @predictmethod
    def predict(self, payload: np.ndarray) -> np.ndarray:
        self.counter += 1
        res1 = sklearn_model(payload)
        
        if res1[0][0] > 0.7:
            return res1
        else:
            return xgboost_model(payload)
        

    def get_counter(self):
        return self.counter
```


```python
sklearn_model.download()
xgboost_model.download()
```


```python
myp = MyPipeline()
```

## Run on Docker


```python
myp.deploy()
```


```python
myp.wait_ready()
```


```python
myp.predict(np.array([[4.9, 3.1, 1.5, 0.2]]))
```


```python
myp.undeploy()
```

## Run On Kubernetes


```python
myp.set_runtime(SeldonKubernetesRuntime())
```


```python
myp.deploy()
```


```python
myp.wait_ready()
```


```python
myp.predict(np.array([[4.9, 3.1, 1.5, 0.2]]))
```


```python
myp.undeploy()
```


```python
yaml = myp.to_k8s_yaml()
print (eval(pprint.pformat(yaml)))
```


```python

```
