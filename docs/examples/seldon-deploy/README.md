# Tempo with Seldon Deploy


## Prerequisites

This notebooks needs to be run in the `tempo-examples` conda environment defined below. Create from project root folder:

```bash
conda env create --name tempo-examples --file conda/tempo-examples.yaml
```

## Project Structure


```python
!tree -P "*.py"  -I "__init__.py|__pycache__" -L 2
```

## Train Model


```python
import os
import logging
import numpy as np
import json
import tempo

from tempo.utils import logger

from src.constants import ARTIFACTS_FOLDER

logger.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
```


```python
from src.data import AdultData

data = AdultData()
```


```python
from src.model import train_model

adult_model = train_model(ARTIFACTS_FOLDER, data)
```

    Train accuracy:  0.9656333333333333
    Test accuracy:  0.854296875


## Create Tempo Artifacts



```python
from src.tempo import create_adult_model

sklearn_model = create_adult_model()
```


```python
# %load src/tempo.py
import os

from src.constants import ARTIFACTS_FOLDER, MODEL_FOLDER

from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model


def create_adult_model() -> Model:
    sklearn_model = Model(
        name="income-sklearn2",
        platform=ModelFramework.SKLearn,
        local_folder=os.path.join(ARTIFACTS_FOLDER, MODEL_FOLDER),
        uri="s3://seldon-models/income/model2",
    )

    return sklearn_model



```

## Save Model



```python
tempo.save(sklearn_model)
```

## Test Locally on Docker

Here we test our models using production images but running locally on Docker. This allows us to ensure the final production deployed model will behave as expected when deployed.


```python
from tempo.seldon import SeldonDockerRuntime

docker_runtime = SeldonDockerRuntime()
docker_runtime.deploy(sklearn_model)
docker_runtime.wait_ready(sklearn_model)
```


```python
r = sklearn_model(data.X_test[0:1])
print(r)
```


```python
docker_runtime.undeploy(sklearn_model)
```

## Deploy With Seldon Deploy

 * Here we illustrate how to run the final models in "production" on Kubernetes by using Tempo to deploy
 
### Prerequisites
 
 A cluster running Seldon deploy.


```python
DEPLOY_USER="admin@seldon.io"
DEPLOY_PASSWORD="12341234"
DEPLOY_INGRESS="35.242.178.205"
```


```python
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```


```python
!kubectl apply -f k8s/rbac -n production
```

    secret/minio-secret configured
    serviceaccount/tempo-pipeline unchanged
    role.rbac.authorization.k8s.io/tempo-pipeline unchanged
    rolebinding.rbac.authorization.k8s.io/tempo-pipeline-rolebinding unchanged


## TODO 
update to create rclone.conf if possible as is hidden behind keycload


```python
from tempo.conf import settings
settings.rclone_cfg = os.getcwd()+"/rclone.conf"
tempo.upload(sklearn_model)
```


```python
from tempo.seldon.deploy import SeldonDeployAuthType, SeldonDeployConfig, SeldonDeployRuntime
from tempo.serve.metadata import RuntimeOptions, KubernetesOptions, IngressOptions

options = RuntimeOptions(
        runtime="tempo.seldon.SeldonKubernetesRuntime",
        k8s_options=KubernetesOptions(namespace="production",
                                     authSecretName="minio-secret"),
        ingress_options=IngressOptions(ssl=True, verify_ssl=False),
)
config = SeldonDeployConfig(
        host=f"https://{DEPLOY_INGRESS}/seldon-deploy/api/v1alpha1",
        user=f"{DEPLOY_USER}",
        password=f"{DEPLOY_PASSWORD}",
        oidc_server=f"https://{DEPLOY_INGRESS}/auth/realms/deploy-realm",
        oidc_client_id="sd-api",
        verify_ssl=False,
        auth_type=SeldonDeployAuthType.oidc,
)

rt = SeldonDeployRuntime(options)
rt.authenticate(settings=config)
```


```python
rt.register(sklearn_model)
```

    {}



```python
rt.deploy(sklearn_model)
rt.wait_ready(sklearn_model)
```

    {'email': 'admin@seldon.io', 'groups': None, 'id': 'seldon', 'name': ''}



    ---------------------------------------------------------------------------

    ApiException                              Traceback (most recent call last)

    <ipython-input-18-5064ca7ca270> in <module>
          1 rt.deploy(sklearn_model)
    ----> 2 rt.wait_ready(sklearn_model)
    

    ~/work/mlops/fork-tempo/tempo/serve/runtime.py in wait_ready(self, model, timeout_secs)
         56         t = model.get_tempo()
         57         t.set_runtime_options_override(self.runtime_options)
    ---> 58         t.wait_ready(self, timeout_secs)
         59 
         60     def to_k8s_yaml(self, model: Any):


    ~/work/mlops/fork-tempo/tempo/serve/base.py in wait_ready(self, runtime, timeout_secs)
        211 
        212     def wait_ready(self, runtime: Runtime, timeout_secs=None):
    --> 213         return runtime.wait_ready_spec(self._get_model_spec(), timeout_secs=timeout_secs)
        214 
        215     def get_endpoint(self, runtime: Runtime):


    ~/work/mlops/fork-tempo/tempo/seldon/deploy.py in wait_ready_spec(self, model_spec, timeout_secs)
        124         while not ready:
        125             sdep: SeldonDeployment = dep_instance.read_seldon_deployment(
    --> 126                 model_spec.model_details.name, model_spec.runtime_options.k8s_options.namespace
        127             )
        128             sdep_dict = sdep.to_dict()


    ~/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/seldon_deploy_sdk/api/seldon_deployments_api.py in read_seldon_deployment(self, name, namespace, **kwargs)
        383             return self.read_seldon_deployment_with_http_info(name, namespace, **kwargs)  # noqa: E501
        384         else:
    --> 385             (data) = self.read_seldon_deployment_with_http_info(name, namespace, **kwargs)  # noqa: E501
        386             return data
        387 


    ~/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/seldon_deploy_sdk/api/seldon_deployments_api.py in read_seldon_deployment_with_http_info(self, name, namespace, **kwargs)
        468             _preload_content=params.get('_preload_content', True),
        469             _request_timeout=params.get('_request_timeout'),
    --> 470             collection_formats=collection_formats)
        471 
        472     def update_seldon_deployment(self, name, namespace, mldeployment, **kwargs):  # noqa: E501


    ~/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/seldon_deploy_sdk/api_client.py in call_api(self, resource_path, method, path_params, query_params, header_params, body, post_params, files, response_type, auth_settings, async_req, _return_http_data_only, collection_formats, _preload_content, _request_timeout)
        355                                    response_type, auth_settings,
        356                                    _return_http_data_only, collection_formats,
    --> 357                                    _preload_content, _request_timeout)
        358         else:
        359             thread = self.pool.apply_async(self.__call_api_with_retry, (resource_path,


    ~/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/seldon_deploy_sdk/api_client.py in __call_api_with_retry(self, resource_path, method, path_params, query_params, header_params, body, post_params, files, response_type, auth_settings, _return_http_data_only, collection_formats, _preload_content, _request_timeout)
        125                             _preload_content=_preload_content,_request_timeout=_request_timeout)
        126             else:
    --> 127                 raise e
        128 
        129     def __call_api(


    ~/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/seldon_deploy_sdk/api_client.py in __call_api_with_retry(self, resource_path, method, path_params, query_params, header_params, body, post_params, files, response_type, auth_settings, _return_http_data_only, collection_formats, _preload_content, _request_timeout)
        113                             response_type=response_type, auth_settings=auth_settings,
        114                             _return_http_data_only=_return_http_data_only, collection_formats=collection_formats,
    --> 115                             _preload_content=_preload_content,_request_timeout=_request_timeout)
        116         except seldon_deploy_sdk.rest.ApiException as e:
        117             #if unauthenticated and have authenticator try refreshing in case token expired


    ~/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/seldon_deploy_sdk/api_client.py in __call_api(self, resource_path, method, path_params, query_params, header_params, body, post_params, files, response_type, auth_settings, _return_http_data_only, collection_formats, _preload_content, _request_timeout)
        186             post_params=post_params, body=body,
        187             _preload_content=_preload_content,
    --> 188             _request_timeout=_request_timeout)
        189 
        190         self.last_response = response_data


    ~/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/seldon_deploy_sdk/api_client.py in request(self, method, url, query_params, headers, post_params, body, _preload_content, _request_timeout)
        376                                         _preload_content=_preload_content,
        377                                         _request_timeout=_request_timeout,
    --> 378                                         headers=headers)
        379         elif method == "HEAD":
        380             return self.rest_client.HEAD(url,


    ~/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/seldon_deploy_sdk/rest.py in GET(self, url, headers, query_params, _preload_content, _request_timeout)
        236                             _preload_content=_preload_content,
        237                             _request_timeout=_request_timeout,
    --> 238                             query_params=query_params)
        239 
        240     def HEAD(self, url, headers=None, query_params=None, _preload_content=True,


    ~/anaconda3/envs/tempo-examples/lib/python3.7/site-packages/seldon_deploy_sdk/rest.py in request(self, method, url, query_params, headers, body, post_params, _preload_content, _request_timeout)
        226 
        227         if not 200 <= r.status <= 299:
    --> 228             raise ApiException(http_resp=r)
        229 
        230         return r


    ApiException: (401)
    Reason: Unauthorized
    HTTP response headers: HTTPHeaderDict({'content-type': 'application/json', 'date': 'Fri, 04 Jun 2021 10:14:51 GMT', 'content-length': '61', 'x-envoy-upstream-service-time': '1', 'server': 'istio-envoy'})
    HTTP response body: {"body":{"code":401,"message":"Unauthorized","requestId":""}}




```python
sklearn_model.remote(data.X_test[0:20])
```


```python
rt.undeploy(sklearn_model)
```

## Production Option 2 (Gitops)

 * We create yaml to provide to our DevOps team to deploy to a production cluster
 * We add Kustomize patches to modify the base Kubernetes yaml created by Tempo


```python
from tempo.seldon.k8s import SeldonKubernetesRuntime

k8s_runtime = SeldonKubernetesRuntime(runtime_options)
yaml_str = k8s_runtime.to_k8s_yaml(explainer)

with open(os.getcwd()+"/k8s/tempo.yaml","w") as f:
    f.write(yaml_str)
```


```python
!kustomize build k8s
```


```python

```
