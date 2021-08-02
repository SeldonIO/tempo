# Sending Insights to Remote Server

This example walks you through the API for sending insights to remote metrics servers.

## Prerequisites

This notebooks needs to be run in the `tempo-examples` conda environment defined below. Create from project root folder:

```bash
conda env create --name tempo-examples --file conda/tempo-examples.yaml
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

ARTIFACTS_FOLDER = os.getcwd()+"/artifacts"
TEMPO_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
```

### Custom inference logic 

Our custom model will be very simple to focus the logic on the insights functionality.



```python
from tempo.serve.metadata import SeldonCoreOptions
from tempo.serve.constants import DefaultInsightsLocalEndpoint, DefaultInsightsK8sEndpoint

seldon_options = SeldonCoreOptions(**{
    "local_options": { "insights_options": { "worker_endpoint": DefaultInsightsLocalEndpoint } },
    "remote_options": {
        "insights_options": { "worker_endpoint": DefaultInsightsK8sEndpoint },
        "namespace": "seldon",
        "authSecretName": "minio-secret",
    },
})
```


```python
import numpy as np
from tempo.serve.utils import pipeline, predictmethod
from tempo.magic import t

@pipeline(
    name='insights-pipeline',
    uri="s3://tempo/insights-pipeline/resources",
    local_folder=ARTIFACTS_FOLDER,
    runtime_options=seldon_options.local_options,
)
class Pipeline:
    
    @predictmethod
    def predict(self, data: np.ndarray, parameters: dict) -> np.ndarray:
        if parameters.get("log"):
            t.insights.log_request()
            t.insights.log_response()
            t.insights.log(parameters)
        return data

```

## Create pipeline


```python
pipeline = Pipeline()
```

## Deploy Docker Insights Dumper


```python
from tempo.docker.utils import deploy_insights_message_dumper

deploy_insights_message_dumper()
```

## Print logs to make sure there's none


```python
from tempo.docker.utils import get_logs_insights_message_dumper

print(get_logs_insights_message_dumper())
```

    


### Explicitly Log Insights 
We explicitly request to log by passing the parameters


```python
params = { "log": "value" }
data = np.array([63])
pred = pipeline(data, params)
print(pred)
```

    Attempted to log request but called manager directly, see documentation [TODO]
    Attempted to log response but called manager directly, see documentation [TODO]


    [63]


## Check the logs

We can see that only the params has been logged. 

This is because even we are passing an input to our model, there is still no request or response. 

To log HTTP request/response, this is relevant when the model is deployed.


```python
print(get_logs_insights_message_dumper())
```

    -----------------
    {
        "path": "/",
        "headers": {
            "host": "0.0.0.0:8080",
            "ce-id": "f7ad4230-c4b4-4022-aae8-7f8cf0c1f8ee",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.NOTIMPLEMENTED.NOTIMPLEMENTED",
            "ce-type": "io.seldon.serving.inference.custominsight",
            "ce-requestid": "f7ad4230-c4b4-4022-aae8-7f8cf0c1f8ee",
            "ce-modelid": "NOTIMPLEMENTED",
            "ce-inferenceservicename": "NOTIMPLEMENTED",
            "ce-namespace": "NOTIMPLEMENTED",
            "ce-endpoint": "NOTIMPLEMENTED",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "16",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"log\": \"value\"}",
        "fresh": false,
        "hostname": "0.0.0.0",
        "ip": "::ffff:172.19.0.1",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "5229eb0f11ce"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.19.0.1 - - [30/Jul/2021:14:43:53 +0000] "POST / HTTP/1.1" 200 1000 "-" "Python/3.7 aiohttp/3.7.4.post0"
    


### Don't log insights

We are now going to send the values with the blank parameters to avoid passing through the branch that explicitly logs the insights.


```python
params = { }
data = np.array([63])
pred = pipeline(data, params)
print(pred)
```

    [63]


## Check logs again
We can see that no new logs have been added


```python
print(get_logs_insights_message_dumper())
```

    -----------------
    {
        "path": "/",
        "headers": {
            "host": "0.0.0.0:8080",
            "ce-id": "f7ad4230-c4b4-4022-aae8-7f8cf0c1f8ee",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.NOTIMPLEMENTED.NOTIMPLEMENTED",
            "ce-type": "io.seldon.serving.inference.custominsight",
            "ce-requestid": "f7ad4230-c4b4-4022-aae8-7f8cf0c1f8ee",
            "ce-modelid": "NOTIMPLEMENTED",
            "ce-inferenceservicename": "NOTIMPLEMENTED",
            "ce-namespace": "NOTIMPLEMENTED",
            "ce-endpoint": "NOTIMPLEMENTED",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "16",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"log\": \"value\"}",
        "fresh": false,
        "hostname": "0.0.0.0",
        "ip": "::ffff:172.19.0.1",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "5229eb0f11ce"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.19.0.1 - - [30/Jul/2021:14:43:53 +0000] "POST / HTTP/1.1" 200 1000 "-" "Python/3.7 aiohttp/3.7.4.post0"
    


### Deploy the  Model to Docker

Let's save the model specifying our depenencies in a conda.yaml.

We'll deploy first to Docker to test.


```python
!ls artifacts/conda.yaml
```

    artifacts/conda.yaml



```python
from tempo.serve.loader import save
save(pipeline, save_env=True)
```

    Collecting packages...
    Packing environment at '/home/clive/anaconda3/envs/tempo-213a907e-f213-4c52-8f69-0eaeb0a0a132' to '/home/clive/work/mlops/fork-tempo/docs/examples/logging-insights/artifacts/environment.tar.gz'
    [########################################] | 100% Completed | 12.6s



```python
from tempo.serve.constants import DefaultInsightsDockerEndpoint
from tempo import deploy_local

seldon_options.local_options.insights_options.worker_endpoint = DefaultInsightsDockerEndpoint

remote_model = deploy_local(pipeline, seldon_options)
```

We can now test our model deployed in Docker as:

## Log insights


```python
params = { "log": "value" }
data = np.array([63])
remote_model.predict(data=data, parameters=params)
```




    array([63])



## Check that all logs are now present
Now we can see that in our model running in docker, the request and response were also logged as per our code logic


```python
print(get_logs_insights_message_dumper())
```

    -----------------
    {
        "path": "/",
        "headers": {
            "host": "0.0.0.0:8080",
            "ce-id": "f7ad4230-c4b4-4022-aae8-7f8cf0c1f8ee",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.NOTIMPLEMENTED.NOTIMPLEMENTED",
            "ce-type": "io.seldon.serving.inference.custominsight",
            "ce-requestid": "f7ad4230-c4b4-4022-aae8-7f8cf0c1f8ee",
            "ce-modelid": "NOTIMPLEMENTED",
            "ce-inferenceservicename": "NOTIMPLEMENTED",
            "ce-namespace": "NOTIMPLEMENTED",
            "ce-endpoint": "NOTIMPLEMENTED",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "16",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"log\": \"value\"}",
        "fresh": false,
        "hostname": "0.0.0.0",
        "ip": "::ffff:172.19.0.1",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "5229eb0f11ce"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.19.0.1 - - [30/Jul/2021:14:43:53 +0000] "POST / HTTP/1.1" 200 1000 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
            "ce-id": "2117cf59-2750-44af-aadf-06262f559d64",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.NOTIMPLEMENTED.NOTIMPLEMENTED",
            "ce-type": "io.seldon.serving.inference.custominsight",
            "ce-requestid": "2117cf59-2750-44af-aadf-06262f559d64",
            "ce-modelid": "NOTIMPLEMENTED",
            "ce-inferenceservicename": "NOTIMPLEMENTED",
            "ce-namespace": "NOTIMPLEMENTED",
            "ce-endpoint": "NOTIMPLEMENTED",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "16",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"log\": \"value\"}",
        "fresh": false,
        "hostname": "insights-dumper",
        "ip": "::ffff:172.19.0.3",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "5229eb0f11ce"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.19.0.3 - - [30/Jul/2021:14:54:37 +0000] "POST / HTTP/1.1" 200 1016 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
            "ce-id": "2117cf59-2750-44af-aadf-06262f559d64",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.NOTIMPLEMENTED.NOTIMPLEMENTED",
            "ce-type": "io.seldon.serving.inference.request",
            "ce-requestid": "2117cf59-2750-44af-aadf-06262f559d64",
            "ce-modelid": "NOTIMPLEMENTED",
            "ce-inferenceservicename": "NOTIMPLEMENTED",
            "ce-namespace": "NOTIMPLEMENTED",
            "ce-endpoint": "NOTIMPLEMENTED",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "345",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"id\": \"2117cf59-2750-44af-aadf-06262f559d64\", \"parameters\": null, \"inputs\": [{\"name\": \"data\", \"shape\": [1], \"datatype\": \"INT64\", \"parameters\": null, \"data\": [63]}, {\"name\": \"parameters\", \"shape\": [16], \"datatype\": \"BYTES\", \"parameters\": null, \"data\": [123, 39, 108, 111, 103, 39, 58, 32, 39, 118, 97, 108, 117, 101, 39, 125]}], \"outputs\": null}",
        "fresh": false,
        "hostname": "insights-dumper",
        "ip": "::ffff:172.19.0.3",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "5229eb0f11ce"
        },
        "connection": {},
        "json": {
            "id": "2117cf59-2750-44af-aadf-06262f559d64",
            "parameters": null,
            "inputs": [
                {
                    "name": "data",
                    "shape": [
                        1
                    ],
                    "datatype": "INT64",
                    "parameters": null,
                    "data": [
                        63
                    ]
                },
                {
                    "name": "parameters",
                    "shape": [
                        16
                    ],
                    "datatype": "BYTES",
                    "parameters": null,
                    "data": [
                        123,
                        39,
                        108,
                        111,
                        103,
                        39,
                        58,
                        32,
                        39,
                        118,
                        97,
                        108,
                        117,
                        101,
                        39,
                        125
                    ]
                }
            ],
            "outputs": null
        }
    }
    ::ffff:172.19.0.3 - - [30/Jul/2021:14:54:37 +0000] "POST / HTTP/1.1" 200 2059 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
            "ce-id": "2117cf59-2750-44af-aadf-06262f559d64",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.NOTIMPLEMENTED.NOTIMPLEMENTED",
            "ce-type": "io.seldon.serving.inference.response",
            "ce-requestid": "2117cf59-2750-44af-aadf-06262f559d64",
            "ce-modelid": "NOTIMPLEMENTED",
            "ce-inferenceservicename": "NOTIMPLEMENTED",
            "ce-namespace": "NOTIMPLEMENTED",
            "ce-endpoint": "NOTIMPLEMENTED",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "153",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"model_name\": \"insights-pipeline\", \"outputs\": [{\"name\": \"output0\", \"datatype\": \"INT64\", \"data\": [63], \"shape\": [1]}], \"model_version\": \"NOTIMPLEMENTED\"}",
        "fresh": false,
        "hostname": "insights-dumper",
        "ip": "::ffff:172.19.0.3",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "5229eb0f11ce"
        },
        "connection": {},
        "json": {
            "model_name": "insights-pipeline",
            "outputs": [
                {
                    "name": "output0",
                    "datatype": "INT64",
                    "data": [
                        63
                    ],
                    "shape": [
                        1
                    ]
                }
            ],
            "model_version": "NOTIMPLEMENTED"
        }
    }
    ::ffff:172.19.0.3 - - [30/Jul/2021:14:54:37 +0000] "POST / HTTP/1.1" 200 1404 "-" "Python/3.7 aiohttp/3.7.4.post0"
    


## Don't log


```python
params = { }
data = np.array([63])
remote_model.predict(data=data, parameters=params)
```




    array([63])



## Also we can see logs are not present when requested

When providing the explicit configuration the logs are not sent, and this can be seen in the insights logger container logs


```python
print(get_logs_insights_message_dumper())
```

    -----------------
    {
        "path": "/",
        "headers": {
            "host": "0.0.0.0:8080",
            "ce-id": "3254a0e9-d436-4355-b279-dbf44eb1f7de",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.NOTIMPLEMENTED.NOTIMPLEMENTED",
            "ce-type": "io.seldon.serving.inference.custominsight",
            "ce-requestid": "3254a0e9-d436-4355-b279-dbf44eb1f7de",
            "ce-modelid": "NOTIMPLEMENTED",
            "ce-inferenceservicename": "NOTIMPLEMENTED",
            "ce-namespace": "NOTIMPLEMENTED",
            "ce-endpoint": "NOTIMPLEMENTED",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.6.2",
            "content-length": "16",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"log\": \"value\"}",
        "fresh": false,
        "hostname": "0.0.0.0",
        "ip": "::ffff:172.18.0.1",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "fa4f8912b8f8"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.18.0.1 - - [22/Jul/2021:12:11:46 +0000] "POST / HTTP/1.1" 200 994 "-" "Python/3.7 aiohttp/3.6.2"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
            "ce-id": "335f65fa-3629-4db6-b638-3ac0943c23d6",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.NOTIMPLEMENTED.NOTIMPLEMENTED",
            "ce-type": "io.seldon.serving.inference.custominsight",
            "ce-requestid": "335f65fa-3629-4db6-b638-3ac0943c23d6",
            "ce-modelid": "NOTIMPLEMENTED",
            "ce-inferenceservicename": "NOTIMPLEMENTED",
            "ce-namespace": "NOTIMPLEMENTED",
            "ce-endpoint": "NOTIMPLEMENTED",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "16",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"log\": \"value\"}",
        "fresh": false,
        "hostname": "insights-dumper",
        "ip": "::ffff:172.18.0.3",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "fa4f8912b8f8"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.18.0.3 - - [22/Jul/2021:12:15:17 +0000] "POST / HTTP/1.1" 200 1016 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
            "ce-id": "335f65fa-3629-4db6-b638-3ac0943c23d6",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.NOTIMPLEMENTED.NOTIMPLEMENTED",
            "ce-type": "io.seldon.serving.inference.request",
            "ce-requestid": "335f65fa-3629-4db6-b638-3ac0943c23d6",
            "ce-modelid": "NOTIMPLEMENTED",
            "ce-inferenceservicename": "NOTIMPLEMENTED",
            "ce-namespace": "NOTIMPLEMENTED",
            "ce-endpoint": "NOTIMPLEMENTED",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "345",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"id\": \"335f65fa-3629-4db6-b638-3ac0943c23d6\", \"parameters\": null, \"inputs\": [{\"name\": \"data\", \"shape\": [1], \"datatype\": \"INT64\", \"parameters\": null, \"data\": [63]}, {\"name\": \"parameters\", \"shape\": [16], \"datatype\": \"BYTES\", \"parameters\": null, \"data\": [123, 39, 108, 111, 103, 39, 58, 32, 39, 118, 97, 108, 117, 101, 39, 125]}], \"outputs\": null}",
        "fresh": false,
        "hostname": "insights-dumper",
        "ip": "::ffff:172.18.0.3",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "fa4f8912b8f8"
        },
        "connection": {},
        "json": {
            "id": "335f65fa-3629-4db6-b638-3ac0943c23d6",
            "parameters": null,
            "inputs": [
                {
                    "name": "data",
                    "shape": [
                        1
                    ],
                    "datatype": "INT64",
                    "parameters": null,
                    "data": [
                        63
                    ]
                },
                {
                    "name": "parameters",
                    "shape": [
                        16
                    ],
                    "datatype": "BYTES",
                    "parameters": null,
                    "data": [
                        123,
                        39,
                        108,
                        111,
                        103,
                        39,
                        58,
                        32,
                        39,
                        118,
                        97,
                        108,
                        117,
                        101,
                        39,
                        125
                    ]
                }
            ],
            "outputs": null
        }
    }
    ::ffff:172.18.0.3 - - [22/Jul/2021:12:15:17 +0000] "POST / HTTP/1.1" 200 2059 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
            "ce-id": "335f65fa-3629-4db6-b638-3ac0943c23d6",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.NOTIMPLEMENTED.NOTIMPLEMENTED",
            "ce-type": "io.seldon.serving.inference.response",
            "ce-requestid": "335f65fa-3629-4db6-b638-3ac0943c23d6",
            "ce-modelid": "NOTIMPLEMENTED",
            "ce-inferenceservicename": "NOTIMPLEMENTED",
            "ce-namespace": "NOTIMPLEMENTED",
            "ce-endpoint": "NOTIMPLEMENTED",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "153",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"model_name\": \"insights-pipeline\", \"outputs\": [{\"name\": \"output0\", \"datatype\": \"INT64\", \"data\": [63], \"shape\": [1]}], \"model_version\": \"NOTIMPLEMENTED\"}",
        "fresh": false,
        "hostname": "insights-dumper",
        "ip": "::ffff:172.18.0.3",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "fa4f8912b8f8"
        },
        "connection": {},
        "json": {
            "model_name": "insights-pipeline",
            "outputs": [
                {
                    "name": "output0",
                    "datatype": "INT64",
                    "data": [
                        63
                    ],
                    "shape": [
                        1
                    ]
                }
            ],
            "model_version": "NOTIMPLEMENTED"
        }
    }
    ::ffff:172.18.0.3 - - [22/Jul/2021:12:15:17 +0000] "POST / HTTP/1.1" 200 1404 "-" "Python/3.7 aiohttp/3.7.4.post0"
    


## Now undeploy to move to Kubernetes


```python
remote_model.undeploy()
```


```python
from tempo.docker.utils import undeploy_insights_message_dumper

undeploy_insights_message_dumper()
```

## Deploy to Kubernetes with Tempo

 * Here we illustrate how the same workflow applies in kubernetes
 
### Prerequisites
 
 Create a Kind Kubernetes cluster with Minio and Seldon Core installed using Ansible from the Tempo project Ansible playbook.
 
 ```
 ansible-playbook ansible/playbooks/seldon_core.yaml
 ```


```python
!kubectl create ns seldon
```

    Error from server (AlreadyExists): namespaces "seldon" already exists



```python
!kubectl apply -f k8s/rbac -n seldon
```

    secret/minio-secret created
    serviceaccount/tempo-pipeline created
    role.rbac.authorization.k8s.io/tempo-pipeline created
    rolebinding.rbac.authorization.k8s.io/tempo-pipeline-rolebinding created



```python
from tempo.examples.minio import create_minio_rclone
import os
create_minio_rclone(os.getcwd()+"/rclone.conf")
```


```python
from tempo.serve.loader import upload
upload(pipeline)
```


```python
from tempo.k8s.utils import deploy_insights_message_dumper

deploy_insights_message_dumper()
```


```python
from tempo.k8s.utils import get_logs_insights_message_dumper

print(get_logs_insights_message_dumper())
```

    



```python
from tempo import deploy_remote

remote_model = deploy_remote(pipeline, seldon_options)
```

## Log insights


```python
params = { "log": "value" }
data = np.array([63])
remote_model.predict(data=data, parameters=params)
```




    array([63])




```python
from tempo.k8s.utils import get_logs_insights_message_dumper

print(get_logs_insights_message_dumper())
```

    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper.seldon-system:8080",
            "ce-id": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.insights-pipeline.seldon",
            "ce-type": "io.seldon.serving.inference.custominsight",
            "ce-requestid": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-modelid": "insights-pipeline",
            "ce-inferenceservicename": "insights-pipeline",
            "ce-namespace": "seldon",
            "ce-endpoint": "default",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "16",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"log\": \"value\"}",
        "fresh": false,
        "hostname": "insights-dumper.seldon-system",
        "ip": "::ffff:10.244.0.24",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "insights-dumper"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:10.244.0.24 - - [30/Jul/2021:14:59:14 +0000] "POST / HTTP/1.1" 200 1034 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper.seldon-system:8080",
            "ce-id": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.insights-pipeline.seldon",
            "ce-type": "io.seldon.serving.inference.request",
            "ce-requestid": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-modelid": "insights-pipeline",
            "ce-inferenceservicename": "insights-pipeline",
            "ce-namespace": "seldon",
            "ce-endpoint": "default",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "345",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"id\": \"93295ae4-2a13-4198-bde4-3fecd91cc135\", \"parameters\": null, \"inputs\": [{\"name\": \"data\", \"shape\": [1], \"datatype\": \"INT64\", \"parameters\": null, \"data\": [63]}, {\"name\": \"parameters\", \"shape\": [16], \"datatype\": \"BYTES\", \"parameters\": null, \"data\": [123, 39, 108, 111, 103, 39, 58, 32, 39, 118, 97, 108, 117, 101, 39, 125]}], \"outputs\": null}",
        "fresh": false,
        "hostname": "insights-dumper.seldon-system",
        "ip": "::ffff:10.244.0.24",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "insights-dumper"
        },
        "connection": {},
        "json": {
            "id": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "parameters": null,
            "inputs": [
                {
                    "name": "data",
                    "shape": [
                        1
                    ],
                    "datatype": "INT64",
                    "parameters": null,
                    "data": [
                        63
                    ]
                },
                {
                    "name": "parameters",
                    "shape": [
                        16
                    ],
                    "datatype": "BYTES",
                    "parameters": null,
                    "data": [
                        123,
                        39,
                        108,
                        111,
                        103,
                        39,
                        58,
                        32,
                        39,
                        118,
                        97,
                        108,
                        117,
                        101,
                        39,
                        125
                    ]
                }
            ],
            "outputs": null
        }
    }
    ::ffff:10.244.0.24 - - [30/Jul/2021:14:59:14 +0000] "POST / HTTP/1.1" 200 2077 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper.seldon-system:8080",
            "ce-id": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.insights-pipeline.seldon",
            "ce-type": "io.seldon.serving.inference.response",
            "ce-requestid": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-modelid": "insights-pipeline",
            "ce-inferenceservicename": "insights-pipeline",
            "ce-namespace": "seldon",
            "ce-endpoint": "default",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "153",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"model_name\": \"insights-pipeline\", \"outputs\": [{\"name\": \"output0\", \"datatype\": \"INT64\", \"data\": [63], \"shape\": [1]}], \"model_version\": \"NOTIMPLEMENTED\"}",
        "fresh": false,
        "hostname": "insights-dumper.seldon-system",
        "ip": "::ffff:10.244.0.24",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "insights-dumper"
        },
        "connection": {},
        "json": {
            "model_name": "insights-pipeline",
            "outputs": [
                {
                    "name": "output0",
                    "datatype": "INT64",
                    "data": [
                        63
                    ],
                    "shape": [
                        1
                    ]
                }
            ],
            "model_version": "NOTIMPLEMENTED"
        }
    }
    ::ffff:10.244.0.24 - - [30/Jul/2021:14:59:14 +0000] "POST / HTTP/1.1" 200 1422 "-" "Python/3.7 aiohttp/3.7.4.post0"
    


## Don't log


```python
params = {  }
data = np.array([63])
remote_model.predict(data=data, parameters=params)
```




    array([63])




```python
print(get_logs_insights_message_dumper())
```

    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper.seldon-system:8080",
            "ce-id": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.insights-pipeline.seldon",
            "ce-type": "io.seldon.serving.inference.custominsight",
            "ce-requestid": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-modelid": "insights-pipeline",
            "ce-inferenceservicename": "insights-pipeline",
            "ce-namespace": "seldon",
            "ce-endpoint": "default",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "16",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"log\": \"value\"}",
        "fresh": false,
        "hostname": "insights-dumper.seldon-system",
        "ip": "::ffff:10.244.0.24",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "insights-dumper"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:10.244.0.24 - - [30/Jul/2021:14:59:14 +0000] "POST / HTTP/1.1" 200 1034 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper.seldon-system:8080",
            "ce-id": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.insights-pipeline.seldon",
            "ce-type": "io.seldon.serving.inference.request",
            "ce-requestid": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-modelid": "insights-pipeline",
            "ce-inferenceservicename": "insights-pipeline",
            "ce-namespace": "seldon",
            "ce-endpoint": "default",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "345",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"id\": \"93295ae4-2a13-4198-bde4-3fecd91cc135\", \"parameters\": null, \"inputs\": [{\"name\": \"data\", \"shape\": [1], \"datatype\": \"INT64\", \"parameters\": null, \"data\": [63]}, {\"name\": \"parameters\", \"shape\": [16], \"datatype\": \"BYTES\", \"parameters\": null, \"data\": [123, 39, 108, 111, 103, 39, 58, 32, 39, 118, 97, 108, 117, 101, 39, 125]}], \"outputs\": null}",
        "fresh": false,
        "hostname": "insights-dumper.seldon-system",
        "ip": "::ffff:10.244.0.24",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "insights-dumper"
        },
        "connection": {},
        "json": {
            "id": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "parameters": null,
            "inputs": [
                {
                    "name": "data",
                    "shape": [
                        1
                    ],
                    "datatype": "INT64",
                    "parameters": null,
                    "data": [
                        63
                    ]
                },
                {
                    "name": "parameters",
                    "shape": [
                        16
                    ],
                    "datatype": "BYTES",
                    "parameters": null,
                    "data": [
                        123,
                        39,
                        108,
                        111,
                        103,
                        39,
                        58,
                        32,
                        39,
                        118,
                        97,
                        108,
                        117,
                        101,
                        39,
                        125
                    ]
                }
            ],
            "outputs": null
        }
    }
    ::ffff:10.244.0.24 - - [30/Jul/2021:14:59:14 +0000] "POST / HTTP/1.1" 200 2077 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper.seldon-system:8080",
            "ce-id": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-specversion": "0.3",
            "ce-source": "io.seldon.serving.deployment.insights-pipeline.seldon",
            "ce-type": "io.seldon.serving.inference.response",
            "ce-requestid": "93295ae4-2a13-4198-bde4-3fecd91cc135",
            "ce-modelid": "insights-pipeline",
            "ce-inferenceservicename": "insights-pipeline",
            "ce-namespace": "seldon",
            "ce-endpoint": "default",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "153",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"model_name\": \"insights-pipeline\", \"outputs\": [{\"name\": \"output0\", \"datatype\": \"INT64\", \"data\": [63], \"shape\": [1]}], \"model_version\": \"NOTIMPLEMENTED\"}",
        "fresh": false,
        "hostname": "insights-dumper.seldon-system",
        "ip": "::ffff:10.244.0.24",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "insights-dumper"
        },
        "connection": {},
        "json": {
            "model_name": "insights-pipeline",
            "outputs": [
                {
                    "name": "output0",
                    "datatype": "INT64",
                    "data": [
                        63
                    ],
                    "shape": [
                        1
                    ]
                }
            ],
            "model_version": "NOTIMPLEMENTED"
        }
    }
    ::ffff:10.244.0.24 - - [30/Jul/2021:14:59:14 +0000] "POST / HTTP/1.1" 200 1422 "-" "Python/3.7 aiohttp/3.7.4.post0"
    



```python
remote_model.undeploy()
```


```python
from tempo.k8s.utils import undeploy_insights_message_dumper

undeploy_insights_message_dumper()
```


```python

```
