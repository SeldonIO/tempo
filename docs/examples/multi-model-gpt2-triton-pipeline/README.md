# Tempo GPT2 Triton ONNX Example


## Prerequisites

TODO

## Create Tempo Artifacts

 * Here we create the Tempo models and orchestration Pipeline for our final service using our models.
 * For illustration the final service will call the sklearn model and based on the result will decide to return that prediction or call the xgboost model and return that prediction instead.


```python
!mkdir artifacts/
```

    mkdir: cannot create directory ‘artifacts/’: File exists



```python
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained(
    "gpt2", from_pt=True, pad_token_id=tokenizer.eos_token_id
)
```


```python
model.save_pretrained("./artifacts/gpt2-model", saved_model=True)
tokenizer.save_pretrained("./artifacts/gpt2-transformer")
```


```python
!mkdir -p artifacts/gpt2-onnx-model/gpt2-model/1/
```


```python
!python -m tf2onnx.convert --saved-model ./artifacts/gpt2-model/saved_model/1 --opset 11  --output ./artifacts/gpt2-onnx-model/gpt2-model/1/model.onnx
```


```python
import os

ARTIFACT_FOLDER = os.getcwd() + "/artifacts"
```


```python
import numpy as np

from tempo.serve.metadata import ModelFramework, ModelDataArgs, ModelDataArg
from tempo.serve.model import Model
from tempo.serve.pipeline import Pipeline, PipelineModels
from tempo.serve.utils import pipeline, predictmethod

```


```python
gpt2_model = Model(
    name="gpt2-model",
    platform=ModelFramework.ONNX,
    local_folder=ARTIFACT_FOLDER + "/gpt2-onnx-model",
    uri="s3://tempo/gpt2/model",
    # TODO: Simplify without need to add output types if array by default
    inputs={},
    outputs=(np.ndarray,np.ndarray,),
    description="GPT-2 ONNX Triton Model",
)
```

    INFO:tempo:Initialising Insights Manager with Args: ('', 1, 1, 3, 0)
    WARNING:tempo:Insights Manager not initialised as empty URL provided.



```python
from tempo.serve.deploy import deploy_local

remote_gpt2_model = deploy_local(gpt2_model)
```


```python
input_ids = tokenizer.encode("This is a test", return_tensors="tf")
attention_mask = np.ones(input_ids.shape.as_list(), dtype=np.int32)

gpt2_inputs = {
    "input_ids:0": input_ids.numpy(),
    "attention_mask:0": attention_mask
}

print(gpt2_inputs)

gpt2_outputs = remote_gpt2_model.predict(**gpt2_inputs)
```

    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True


    {'input_ids:0': array([[1212,  318,  257, 1332]], dtype=int32), 'attention_mask:0': array([[1, 1, 1, 1]], dtype=int32)}


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    



```python
logits = gpt2_outputs["logits"]

# take the best next token probability of the last token of input ( greedy approach)
next_token = logits.argmax(axis=2)[0]
next_token_str = tokenizer.decode(
    next_token[-1:], skip_special_tokens=True, clean_up_tokenization_spaces=True
).strip()

print(next_token_str)
```

    of



```python
@pipeline(
    name="gpt2-transformer",
    uri="s3://tempo/gpt2/transformer",
    local_folder=ARTIFACT_FOLDER + "/gpt2-transformer/",
    models=PipelineModels(gpt2_model=gpt2_model),
    description="A pipeline to use either an sklearn or xgboost model for Iris classification",
)
class GPT2Transformer:
    # TODO: Set ready = false in init to avoid having to set it
    def __init__(self):
        self.ready = False
        
    # TODO: Bug - Pipeline locally doesn't call the load function (expected?)
    # TODO: Update load function to change ready to true by default
    def load(self, tokenizer_path="/mnt/models/"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)       
        self.ready = True

    @predictmethod
    def predict(self, payload: np.ndarray) -> np.ndarray:
        count = 0
        # TODO: Update to allow this to be passed as parameters
        max_gen_len = 10
        # TODO: Update to work for multiple sentences
        gen_sentence = payload[0]
        while count < max_gen_len:
            input_ids = self.tokenizer.encode(gen_sentence, return_tensors="tf")
            attention_mask = np.ones(input_ids.shape.as_list(), dtype=np.int32)

            gpt2_inputs = {
                "input_ids:0": input_ids.numpy(),
                "attention_mask:0": attention_mask
            }

            gpt2_outputs = remote_gpt2_model.predict(**gpt2_inputs)

            logits = gpt2_outputs["logits"]

            # take the best next token probability of the last token of input ( greedy approach)
            next_token = logits.argmax(axis=2)[0]
            next_token_str = tokenizer.decode(
                next_token[-1:], skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            
            gen_sentence += " " + next_token_str
            count += 1
        
        return gen_sentence

```

    INFO:tempo:Initialising Insights Manager with Args: ('', 1, 1, 3, 0)
    WARNING:tempo:Insights Manager not initialised as empty URL provided.



```python
gpt2_transformer = GPT2Transformer()
# Load locally manually
gpt2_transformer.load(tokenizer_path=ARTIFACT_FOLDER + "/gpt2-transformer")
```


```python
gpt2_output = gpt2_transformer.predict(["I love artificial intelligence"])
```

    DEBUG:tempo:Setting context to context for insights manager
    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    DEBUG:tempo:Using remote class tempo.seldon.SeldonDockerRuntime
    DEBUG:tempo:Calling requests POST with endpoint=http://0.0.0.0:39609/v2/models/gpt2-model/infer headers={} verify=True
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    



```python
print(gpt2_output)
```

    I love artificial intelligence , but I 'm not sure if it 's worth


## Save Classifier Environment

 * In preparation for running our models we save the Python environment needed for the orchestration to run as defined by a `conda.yaml` in our project.


```python
%%writefile artifacts/gpt2-transformer/conda.yaml
name: tempo-gpt2
channels:
  - defaults
dependencies:
  - python=3.7.10
  - pip:
    - transformers==4.5.1
    - tokenizers==0.10.3
    - mlserver
    - mlserver-tempo
```

    Overwriting artifacts/gpt2-transformer/conda.yaml



```python
from tempo.serve.loader import save
save(gpt2_transformer)
```

    INFO:tempo:Initialising Insights Manager with Args: ('', 1, 1, 3, 0)
    WARNING:tempo:Insights Manager not initialised as empty URL provided.
    INFO:tempo:Saving environment
    INFO:tempo:Saving tempo model to /home/alejandro/Programming/kubernetes/seldon/tempo/docs/examples/multi-model-gpt2-triton-pipeline/artifacts/gpt2-transformer/model.pickle
    INFO:tempo:Using found conda.yaml
    INFO:tempo:Creating conda env with: conda env create --name tempo-40d229b4-4ca9-405a-80a1-d336b174add1 --file /tmp/tmpwm6mf7mo.yml


## Test Locally on Docker

 * Here we test our models using production images but running locally on Docker. This allows us to ensure the final production deployed model will behave as expected when deployed.


```python
from tempo import deploy_local
remote_pipeline = deploy_local(gpt2_pipeline)
```


```python
remote_model.predict(["I love artificial intelligence"])
```


```python
remote_model.undeploy()
```

## Production Option 1 (Deploy to Kubernetes with Tempo)

 * Here we illustrate how to run the final models in "production" on Kubernetes by using Tempo to deploy
 
### Prerequisites
 
Create a Kind Kubernetes cluster with Minio and Seldon Core installed using Ansible as described [here](https://tempo.readthedocs.io/en/latest/overview/quickstart.html#kubernetes-cluster-with-seldon-core).


```python
!kubectl apply -f k8s/rbac -n production
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
upload(gpt2_pipeline)
```


```python
from tempo.serve.metadata import SeldonCoreOptions
runtime_options = SeldonCoreOptions(**{
        "remote_options": {
            "namespace": "production",
            "authSecretName": "minio-secret"
        }
    })
```


```python
from tempo import deploy_remote
remote_model = deploy_remote(gpt2_pipeline, options=runtime_options)
```


```python
remote_model.predict(["I love artificial intelligence"])
```

    {'output0': array([1.], dtype=float32), 'output1': 'sklearn prediction'}
    {'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32), 'output1': 'xgboost prediction'}



```python
remote_model.undeploy()
```
