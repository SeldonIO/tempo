# Seldon Deploy Example


```python
from tempo.serve.metadata import ModelFramework, KubernetesOptions
from tempo.serve.model import Model
from tempo.seldon.deploy import SeldonDeployRuntime
import os
import pprint
import numpy as np

rt = SeldonDeployRuntime(host="http://34.105.136.157/seldon-deploy/api/v1alpha1",
                        user="admin@kubeflow.org",
                        password= "12341234",
                        k8s_options=KubernetesOptions(namespace="seldon"))

sklearn_model = Model(
    name="test-iris-sklearn",
    runtime=rt,
    platform=ModelFramework.SKLearn,
    uri="gs://seldon-models/sklearn/iris",
    local_folder=os.getcwd() + "/sklearn")

```


```python
sklearn_model.deploy()
```


```python
sklearn_model.wait_ready()
```


```python
sklearn_model.get_endpoint()
```


```python
sklearn_model(np.array([[4.9, 3.1, 1.5, 0.2]]))
```


```python
sklearn_model.undeploy()
```


```python

```
