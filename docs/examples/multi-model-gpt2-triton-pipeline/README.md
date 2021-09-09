# Tempo GPT2 Triton ONNX Example


## Prerequisites

TODO

### Workflow Overview

In this example we will be doing the following:
* Download & optimize pre-trained artifacts
* Deploy GPT2 Model and Test in Docker
* Deploy GPT2 Pipeline and Test in Docker
* Deploy GPT2 Pipeline & Model to Kuberntes and Test

### Install Dependencies


```python
%%writefile requirements-dev.txt
transformers==4.5.1
torch==1.8.1
tokenizers==0.10.3
tensorflow==2.4.1
tf2onnx==1.8.5
```


```python
pip install -r requirements-dev.txt
```

## Download & Optimize pre-trained artifacts


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

    All PyTorch model weights were used when initializing TFGPT2LMHeadModel.
    
    All the weights of TFGPT2LMHeadModel were initialized from the PyTorch model.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2LMHeadModel for predictions without further training.



```python
model.save_pretrained("./artifacts/gpt2-model", saved_model=True)
tokenizer.save_pretrained("./artifacts/gpt2-transformer")
```

    WARNING:tensorflow:The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    WARNING:tensorflow:AutoGraph could not transform <bound method Socket.send of <zmq.sugar.socket.Socket object at 0x7faebec9aad0>> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: module, class, method, function, traceback, frame, or code object was expected, got cython_function_or_method
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    WARNING: AutoGraph could not transform <bound method Socket.send of <zmq.sugar.socket.Socket object at 0x7faebec9aad0>> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: module, class, method, function, traceback, frame, or code object was expected, got cython_function_or_method
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    WARNING:tensorflow:The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    WARNING:tensorflow:The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    WARNING:tensorflow:The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    WARNING:tensorflow:The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    WARNING:tensorflow:The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    WARNING:tensorflow:The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    WARNING:tensorflow:The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    WARNING:tensorflow:The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    WARNING:tensorflow:The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    WARNING:tensorflow:The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    WARNING:tensorflow:The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    WARNING:tensorflow:The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    WARNING:tensorflow:The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
    WARNING:tensorflow:The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
    WARNING:tensorflow:The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.


    WARNING:absl:Found untraced functions such as wte_layer_call_and_return_conditional_losses, wte_layer_call_fn, dropout_layer_call_and_return_conditional_losses, dropout_layer_call_fn, ln_f_layer_call_and_return_conditional_losses while saving (showing 5 of 735). These functions will not be directly callable after loading.
    WARNING:absl:Found untraced functions such as wte_layer_call_and_return_conditional_losses, wte_layer_call_fn, dropout_layer_call_and_return_conditional_losses, dropout_layer_call_fn, ln_f_layer_call_and_return_conditional_losses while saving (showing 5 of 735). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: ./artifacts/gpt2-model/saved_model/1/assets


    INFO:tensorflow:Assets written to: ./artifacts/gpt2-model/saved_model/1/assets





    ('./artifacts/gpt2-transformer/tokenizer_config.json',
     './artifacts/gpt2-transformer/special_tokens_map.json',
     './artifacts/gpt2-transformer/vocab.json',
     './artifacts/gpt2-transformer/merges.txt',
     './artifacts/gpt2-transformer/added_tokens.json')




```python
!mkdir -p artifacts/gpt2-onnx-model/gpt2-model/1/
```


```python
!python -m tf2onnx.convert --saved-model ./artifacts/gpt2-model/saved_model/1 --opset 11  --output ./artifacts/gpt2-onnx-model/gpt2-model/1/model.onnx
```

    2021-09-07 08:43:11.186716: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    /home/alejandro/miniconda3/lib/python3.7/runpy.py:125: RuntimeWarning: 'tf2onnx.convert' found in sys.modules after import of package 'tf2onnx', but prior to execution of 'tf2onnx.convert'; this may result in unpredictable behaviour
      warn(RuntimeWarning(msg))
    2021-09-07 08:43:12.886148: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
    2021-09-07 08:43:12.886345: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2021-09-07 08:43:12.886376: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
    2021-09-07 08:43:12.886392: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (DESKTOP-CSLUJOT): /proc/driver/nvidia/version does not exist
    2021-09-07 08:43:12.888970: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
    2021-09-07 08:43:12,892 - WARNING - '--tag' not specified for saved_model. Using --tag serve
    2021-09-07 08:43:19,256 - INFO - Signatures found in model: [serving_default].
    2021-09-07 08:43:19,256 - WARNING - '--signature_def' not specified, using first signature: serving_default
    2021-09-07 08:43:19,256 - INFO - Output names: ['logits', 'past_key_values']
    2021-09-07 08:43:19.306853: I tensorflow/core/grappler/devices.cc:69] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
    2021-09-07 08:43:19.307167: I tensorflow/core/grappler/clusters/single_machine.cc:356] Starting new session
    2021-09-07 08:43:19.307630: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
    2021-09-07 08:43:19.316142: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2400005000 Hz
    2021-09-07 08:43:19.501937: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:928] Optimization results for grappler item: graph_to_optimize
      function_optimizer: Graph size after: 3213 nodes (3060), 4128 edges (3974), time = 99.635ms.
      function_optimizer: function_optimizer did nothing. time = 1.408ms.
    
    2021-09-07 08:43:27.473999: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
    WARNING:tensorflow:From /home/alejandro/miniconda3/lib/python3.7/site-packages/tf2onnx/tf_loader.py:603: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.compat.v1.graph_util.extract_sub_graph`
    2021-09-07 08:43:29,277 - WARNING - From /home/alejandro/miniconda3/lib/python3.7/site-packages/tf2onnx/tf_loader.py:603: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.compat.v1.graph_util.extract_sub_graph`
    2021-09-07 08:43:29.353446: I tensorflow/core/grappler/devices.cc:69] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
    2021-09-07 08:43:29.353640: I tensorflow/core/grappler/clusters/single_machine.cc:356] Starting new session
    2021-09-07 08:43:29.353974: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
    2021-09-07 08:43:36.123024: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:928] Optimization results for grappler item: graph_to_optimize
      constant_folding: Graph size after: 2720 nodes (-318), 3646 edges (-319), time = 4489.73486ms.
      function_optimizer: function_optimizer did nothing. time = 2.24ms.
      constant_folding: Graph size after: 2720 nodes (0), 3646 edges (0), time = 1504.76294ms.
      function_optimizer: function_optimizer did nothing. time = 13.628ms.
    
    2021-09-07 08:43:39,251 - INFO - Using tensorflow=2.4.0, onnx=1.9.0, tf2onnx=1.8.5/50049d
    2021-09-07 08:43:39,252 - INFO - Using opset <onnx, 11>
    2021-09-07 08:43:51,608 - INFO - Computed 0 values for constant folding
    2021-09-07 08:44:27,844 - INFO - Optimizing ONNX model
    2021-09-07 08:44:38,170 - INFO - After optimization: Cast -123 (311->188), Concat -37 (126->89), Const -1854 (2032->178), Gather +12 (2->14), GlobalAveragePool +50 (0->50), Identity -76 (76->0), ReduceMean -50 (50->0), Shape -37 (112->75), Slice -74 (235->161), Squeeze -198 (223->25), Transpose -12 (61->49), Unsqueeze -361 (435->74)
    2021-09-07 08:44:39,069 - INFO - 
    2021-09-07 08:44:39,069 - INFO - Successfully converted TensorFlow model ./artifacts/gpt2-model/saved_model/1 to ONNX
    2021-09-07 08:44:39,069 - INFO - Model inputs: ['attention_mask:0', 'input_ids:0']
    2021-09-07 08:44:39,070 - INFO - Model outputs: ['logits', 'past_key_values']
    2021-09-07 08:44:39,070 - INFO - ONNX model is saved at ./artifacts/gpt2-onnx-model/gpt2-model/1/model.onnx


## Deploy GPT2 ONNX Model in Triton


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

#### Define as tempo model


```python
gpt2_model = Model(
    name="gpt2-model",
    platform=ModelFramework.ONNX,
    local_folder=ARTIFACT_FOLDER + "/gpt2-onnx-model",
    uri="s3://tempo/gpt2/model",
    description="GPT-2 ONNX Triton Model",
)
```

    Insights Manager not initialised as empty URL provided.


#### Deploy gpt2 model to docker


```python
from tempo.serve.deploy import deploy_local

remote_gpt2_model = deploy_local(gpt2_model)
```

#### Send predictions


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

    {'input_ids:0': array([[1212,  318,  257, 1332]], dtype=int32), 'attention_mask:0': array([[1, 1, 1, 1]], dtype=int32)}


#### Print single next token generated


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


## Define Transformer Pipeline


```python

@pipeline(
    name="gpt2-transformer",
    uri="s3://tempo/gpt2/transformer",
    local_folder=ARTIFACT_FOLDER + "/gpt2-transformer/",
    models=PipelineModels(gpt2_model=gpt2_model),
    description="A pipeline to use either an sklearn or xgboost model for Iris classification",
)
class GPT2Transformer:
    def __init__(self):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("/mnt/models/")
        except:
            self.tokenizer = GPT2Tokenizer.from_pretrained(ARTIFACT_FOLDER + "/gpt2-transformer/")
        
    @predictmethod
    def predict(self, payload: str) -> str:
        count = 0
        # TODO: Update to allow this to be passed as parameters
        max_gen_len = 10
        # TODO: Update to work for multiple sentences
        gen_sentence = payload
        while count < max_gen_len:
            input_ids = self.tokenizer.encode(gen_sentence, return_tensors="tf")
            attention_mask = np.ones(input_ids.shape.as_list(), dtype=np.int32)

            gpt2_inputs = {
                "input_ids:0": input_ids.numpy(),
                "attention_mask:0": attention_mask
            }

            gpt2_outputs = self.models.gpt2_model.predict(**gpt2_inputs)

            logits = gpt2_outputs["logits"]

            # take the best next token probability of the last token of input ( greedy approach)
            next_token = logits.argmax(axis=2)[0]
            next_token_str = self.tokenizer.decode(
                next_token[-1:], skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            
            gen_sentence += " " + next_token_str
            count += 1
        
        return gen_sentence

```

    INFO:tempo:Initialising Insights Manager with Args: ('', 1, 1, 3, 0)
    WARNING:tempo:Insights Manager not initialised as empty URL provided.


#### Test locally against deployed model


```python
gpt2_transformer = GPT2Transformer()
```


```python
gpt2_output = gpt2_transformer.predict("I love artificial intelligence")
```


```python
print(gpt2_output)
```

    I love artificial intelligence , but I 'm not sure if it 's worth


## Deploy GPT2 Transformer to Docker and Test

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
    - tensorflow==2.4.1
    - dill
    - mlops-tempo
    - mlserver
    - mlserver-tempo
```

    Overwriting artifacts/gpt2-transformer/conda.yaml


#### Save environment and pipeline artifact


```python
from tempo.serve.loader import save
save(GPT2Transformer)
```

    INFO:tempo:Initialising Insights Manager with Args: ('', 1, 1, 3, 0)
    WARNING:tempo:Insights Manager not initialised as empty URL provided.
    INFO:tempo:Saving environment
    INFO:tempo:Saving tempo model to /home/alejandro/Programming/kubernetes/seldon/tempo/docs/examples/multi-model-gpt2-triton-pipeline/artifacts/gpt2-transformer/model.pickle
    INFO:tempo:Using found conda.yaml
    INFO:tempo:Creating conda env with: conda env create --name tempo-cb69ce65-9d45-4683-bdfd-592f735994f1 --file /tmp/tmp1vsizgk7.yml
    INFO:tempo:packing conda environment from tempo-cb69ce65-9d45-4683-bdfd-592f735994f1


    Collecting packages...
    Packing environment at '/home/alejandro/miniconda3/envs/tempo-cb69ce65-9d45-4683-bdfd-592f735994f1' to '/home/alejandro/Programming/kubernetes/seldon/tempo/docs/examples/multi-model-gpt2-triton-pipeline/artifacts/gpt2-transformer/environment.tar.gz'
    [########################################] | 100% Completed | 49.2s


    INFO:tempo:Removing conda env with: conda remove --name tempo-cb69ce65-9d45-4683-bdfd-592f735994f1 --all --yes


#### Deploy locally on Docker

 * Here we test our models using production images but running locally on Docker. This allows us to ensure the final production deployed model will behave as expected when deployed.


```python
from tempo import deploy_local
remote_transformer = deploy_local(gpt2_transformer)
```


```python
remote_transformer.predict("I love artificial intelligence")
```




    "I love artificial intelligence , but I 'm not sure if it 's worth"




```python
remote_transformer.undeploy()
```

    INFO:tempo:Undeploying gpt2-transformer
    INFO:tempo:Undeploying gpt2-model


## Deploy to Kubernetes

 * Here we illustrate how to run the final models in "production" on Kubernetes by using Tempo to deploy
 
### Prerequisites
 
Create a Kind Kubernetes cluster with Minio and Seldon Core installed using Ansible as described [here](https://tempo.readthedocs.io/en/latest/overview/quickstart.html#kubernetes-cluster-with-seldon-core).


```python
!kubectl create ns production
!kubectl apply -f k8s/rbac -n production
```

    Error from server (AlreadyExists): namespaces "production" already exists
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
upload(gpt2_transformer)
upload(gpt2_model)
```

    INFO:tempo:Uploading /home/alejandro/Programming/kubernetes/seldon/tempo/docs/examples/multi-model-gpt2-triton-pipeline/artifacts/gpt2-transformer/ to s3://tempo/gpt2/transformer
    INFO:tempo:Uploading /home/alejandro/Programming/kubernetes/seldon/tempo/docs/examples/multi-model-gpt2-triton-pipeline/artifacts/gpt2-onnx-model to s3://tempo/gpt2/model



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
remote_gpt2_transformer = deploy_remote(gpt2_transformer, options=runtime_options)
```


```python
remote_gpt2_transformer.predict("I love artificial intelligence")
```




    "I love artificial intelligence , but I 'm not sure if it 's worth"




```python
remote_gpt2_transformer.undeploy()
```

    INFO:tempo:Undeploying gpt2-transformer
    INFO:tempo:Undeploying gpt2-model



```python

```
