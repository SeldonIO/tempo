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
import numpy as np
from tempo.serve.utils import pipeline, predictmethod
from tempo.serve.metadata import InsightRequestModes, RuntimeOptions
from tempo.serve.constants import DefaultInsightsLocalEndpoint

from tempo.magic import tempo

local_options = RuntimeOptions(**{"insights_options": {"worker_endpoint": DefaultInsightsLocalEndpoint }})

@pipeline(
    name='insights-pipeline',
    uri="s3://tempo/insights-pipeline/resources",
    local_folder=ARTIFACTS_FOLDER,
    runtime_options=local_options,
)
class Pipeline:
    
    @predictmethod
    def predict(self, data: np.ndarray, parameters: dict) -> np.ndarray:
        if parameters.get("log"):
            tempo.insights.log_request()
            tempo.insights.log_response()
            tempo.insights.log(parameters)
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
            "hostname": "54e1c9dd991e"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.18.0.1 - - [15/Jun/2021:13:12:35 +0000] "POST / HTTP/1.1" 200 553 "-" "Python/3.7 aiohttp/3.6.2"
    


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
We can see that now new logs have been added


```python
print(get_logs_insights_message_dumper())
```

    -----------------
    {
        "path": "/",
        "headers": {
            "host": "0.0.0.0:8080",
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
            "hostname": "54e1c9dd991e"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.18.0.1 - - [15/Jun/2021:13:12:35 +0000] "POST / HTTP/1.1" 200 553 "-" "Python/3.7 aiohttp/3.6.2"
    


### Deploy the  Model to Docker

Finally, we'll be able to deploy our model using Tempo against one of the available runtimes (i.e. Kubernetes, Docker or Seldon Deploy).

We'll deploy first to Docker to test.


```python
%%writetemplate $ARTIFACTS_FOLDER/conda.yaml
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
save(pipeline, save_env=True)
```

    Collecting packages...
    Packing environment at '/home/alejandro/miniconda3/envs/tempo-424eb275-589f-421f-acc3-6dba9d7f803e' to '/home/alejandro/Programming/kubernetes/seldon/tempo/docs/examples/logging-insights/artifacts/environment.tar.gz'
    [########################################] | 100% Completed | 16.1s



```python
from tempo.serve.constants import DefaultInsightsDockerEndpoint
from tempo.serve.metadata import RuntimeOptions
from tempo import deploy

docker_options = RuntimeOptions(**{
    "insights_options": {
        "worker_endpoint": DefaultInsightsDockerEndpoint 
    }
})

remote_model = deploy(pipeline, docker_options)
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
            "hostname": "54e1c9dd991e"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.18.0.1 - - [15/Jun/2021:13:12:35 +0000] "POST / HTTP/1.1" 200 553 "-" "Python/3.7 aiohttp/3.6.2"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
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
            "hostname": "54e1c9dd991e"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.18.0.3 - - [15/Jun/2021:13:16:15 +0000] "POST / HTTP/1.1" 200 575 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "345",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"id\": \"da612e20-63bd-498a-88ff-48f1474c93cc\", \"parameters\": null, \"inputs\": [{\"name\": \"data\", \"shape\": [1], \"datatype\": \"INT64\", \"parameters\": null, \"data\": [63]}, {\"name\": \"parameters\", \"shape\": [16], \"datatype\": \"BYTES\", \"parameters\": null, \"data\": [123, 39, 108, 111, 103, 39, 58, 32, 39, 118, 97, 108, 117, 101, 39, 125]}], \"outputs\": null}",
        "fresh": false,
        "hostname": "insights-dumper",
        "ip": "::ffff:172.18.0.3",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "54e1c9dd991e"
        },
        "connection": {},
        "json": {
            "id": "da612e20-63bd-498a-88ff-48f1474c93cc",
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
    ::ffff:172.18.0.3 - - [15/Jun/2021:13:16:15 +0000] "POST / HTTP/1.1" 200 1624 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "118",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"model_name\": \"insights-pipeline\", \"outputs\": [{\"name\": \"output0\", \"datatype\": \"INT64\", \"data\": [63], \"shape\": [1]}]}",
        "fresh": false,
        "hostname": "insights-dumper",
        "ip": "::ffff:172.18.0.3",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "54e1c9dd991e"
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
            ]
        }
    }
    ::ffff:172.18.0.3 - - [15/Jun/2021:13:16:15 +0000] "POST / HTTP/1.1" 200 890 "-" "Python/3.7 aiohttp/3.7.4.post0"
    


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
            "hostname": "54e1c9dd991e"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.18.0.1 - - [15/Jun/2021:13:12:35 +0000] "POST / HTTP/1.1" 200 553 "-" "Python/3.7 aiohttp/3.6.2"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
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
            "hostname": "54e1c9dd991e"
        },
        "connection": {},
        "json": {
            "log": "value"
        }
    }
    ::ffff:172.18.0.3 - - [15/Jun/2021:13:16:15 +0000] "POST / HTTP/1.1" 200 575 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "345",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"id\": \"da612e20-63bd-498a-88ff-48f1474c93cc\", \"parameters\": null, \"inputs\": [{\"name\": \"data\", \"shape\": [1], \"datatype\": \"INT64\", \"parameters\": null, \"data\": [63]}, {\"name\": \"parameters\", \"shape\": [16], \"datatype\": \"BYTES\", \"parameters\": null, \"data\": [123, 39, 108, 111, 103, 39, 58, 32, 39, 118, 97, 108, 117, 101, 39, 125]}], \"outputs\": null}",
        "fresh": false,
        "hostname": "insights-dumper",
        "ip": "::ffff:172.18.0.3",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "54e1c9dd991e"
        },
        "connection": {},
        "json": {
            "id": "da612e20-63bd-498a-88ff-48f1474c93cc",
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
    ::ffff:172.18.0.3 - - [15/Jun/2021:13:16:15 +0000] "POST / HTTP/1.1" 200 1624 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper:8080",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "118",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"model_name\": \"insights-pipeline\", \"outputs\": [{\"name\": \"output0\", \"datatype\": \"INT64\", \"data\": [63], \"shape\": [1]}]}",
        "fresh": false,
        "hostname": "insights-dumper",
        "ip": "::ffff:172.18.0.3",
        "ips": [],
        "protocol": "http",
        "query": {},
        "subdomains": [],
        "xhr": false,
        "os": {
            "hostname": "54e1c9dd991e"
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
            ]
        }
    }
    ::ffff:172.18.0.3 - - [15/Jun/2021:13:16:15 +0000] "POST / HTTP/1.1" 200 890 "-" "Python/3.7 aiohttp/3.7.4.post0"
    


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
 ansible-playbook ansible/playbooks/default.yaml
 ```


```python
!kubectl apply -f k8s/rbac -n production
```

    secret/minio-secret configured
    serviceaccount/tempo-pipeline unchanged
    role.rbac.authorization.k8s.io/tempo-pipeline unchanged
    rolebinding.rbac.authorization.k8s.io/tempo-pipeline-rolebinding unchanged



```python
from tempo.examples.minio import create_minio_rclone
import os
create_minio_rclone(os.getcwd()+"/rclone.conf")
```


```python
from tempo.serve.loader import upload
upload(pipeline)
```

    2021/06/14 13:04:19 Failed to create file system for "s3://tempo/insights-pipeline/resources": didn't find section in config file
    



```python
from tempo.serve.metadata import RuntimeOptions, KubernetesOptions, InsightsOptions
from tempo.serve.constants import DefaultInsightsK8sEndpoint

runtime_options = RuntimeOptions(
        k8s_options=KubernetesOptions(
            namespace="production",
            authSecretName="minio-secret"
        ),
        insights_options=InsightsOptions(
            worker_endpoint=DefaultInsightsK8sEndpoint
        )
    )
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
from tempo.seldon.k8s import SeldonKubernetesRuntime
k8s_runtime = SeldonKubernetesRuntime(runtime_options)
k8s_runtime.deploy(pipeline)
k8s_runtime.wait_ready(pipeline)
```

## Log insights


```python
params = { "log": "value" }
data = np.array([63])
pipeline.remote(data=data, parameters=params)
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
        "ip": "::ffff:10.1.1.168",
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
    ::ffff:10.1.1.168 - - [14/Jun/2021:12:31:47 +0000] "POST / HTTP/1.1" 200 606 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper.seldon-system:8080",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "345",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"id\": \"8c5c847b-d5ff-460d-a6d7-7c037943f725\", \"parameters\": null, \"inputs\": [{\"name\": \"data\", \"shape\": [1], \"datatype\": \"INT64\", \"parameters\": null, \"data\": [63]}, {\"name\": \"parameters\", \"shape\": [16], \"datatype\": \"BYTES\", \"parameters\": null, \"data\": [123, 39, 108, 111, 103, 39, 58, 32, 39, 118, 97, 108, 117, 101, 39, 125]}], \"outputs\": null}",
        "fresh": false,
        "hostname": "insights-dumper.seldon-system",
        "ip": "::ffff:10.1.1.168",
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
            "id": "8c5c847b-d5ff-460d-a6d7-7c037943f725",
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
    ::ffff:10.1.1.168 - - [14/Jun/2021:12:31:47 +0000] "POST / HTTP/1.1" 200 1655 "-" "Python/3.7 aiohttp/3.7.4.post0"
    


## Don't log


```python
params = {  }
data = np.array([63])
pipeline.remote(data=data, parameters=params)
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
        "ip": "::ffff:10.1.1.168",
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
    ::ffff:10.1.1.168 - - [14/Jun/2021:12:31:47 +0000] "POST / HTTP/1.1" 200 606 "-" "Python/3.7 aiohttp/3.7.4.post0"
    -----------------
    {
        "path": "/",
        "headers": {
            "host": "insights-dumper.seldon-system:8080",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Python/3.7 aiohttp/3.7.4.post0",
            "content-length": "345",
            "content-type": "application/json"
        },
        "method": "POST",
        "body": "{\"id\": \"8c5c847b-d5ff-460d-a6d7-7c037943f725\", \"parameters\": null, \"inputs\": [{\"name\": \"data\", \"shape\": [1], \"datatype\": \"INT64\", \"parameters\": null, \"data\": [63]}, {\"name\": \"parameters\", \"shape\": [16], \"datatype\": \"BYTES\", \"parameters\": null, \"data\": [123, 39, 108, 111, 103, 39, 58, 32, 39, 118, 97, 108, 117, 101, 39, 125]}], \"outputs\": null}",
        "fresh": false,
        "hostname": "insights-dumper.seldon-system",
        "ip": "::ffff:10.1.1.168",
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
            "id": "8c5c847b-d5ff-460d-a6d7-7c037943f725",
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
    ::ffff:10.1.1.168 - - [14/Jun/2021:12:31:47 +0000] "POST / HTTP/1.1" 200 1655 "-" "Python/3.7 aiohttp/3.7.4.post0"
    



```python
k8s_runtime.undeploy(pipeline)
```


```python
from tempo.k8s.utils import undeploy_insights_message_dumper

undeploy_insights_message_dumper()
```


```python

```
