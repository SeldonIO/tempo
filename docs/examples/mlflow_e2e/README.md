# MLflow end-to-end example

In this example we are going to build a model using `mlflow`, pack and deploy locally using `tempo` (in docker and local kubernetes cluster).

We are are going to use follow the MNIST pytorch example from `mlflow`, check this [link](https://github.com/mlflow/mlflow/tree/master/examples/pytorch/MNIST) for more information.



In this example we will:
  * [Train MNIST Model using mlflow and pytorch](#Train-model)
  * [Create tempo artifacts](#Save-model-environment)
  * [Deploy Locally to Docker](#Deploy-to-Docker)
  * [Deploy Locally to Kubernetes](#Deploy-to-Kubernetes)

## Prerequisites

This notebooks needs to be run in the `tempo-examples` conda environment defined below. Create from project root folder:

```bash
conda env create --name tempo-examples --file conda/tempo-examples.yaml
```

## Train model

We train MNIST model below:

### Install prerequisites


```python
!pip install mlflow 'torchvision>=0.9.1' torch==1.9.0 pytorch-lightning==1.4.0
```

    Requirement already satisfied: mlflow in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (1.20.1)
    Requirement already satisfied: torchvision>=0.9.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (0.10.0)
    Requirement already satisfied: torch==1.9.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (1.9.0)
    Requirement already satisfied: pytorch-lightning==1.4.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (1.4.0)
    Requirement already satisfied: tqdm>=4.41.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pytorch-lightning==1.4.0) (4.62.1)
    Requirement already satisfied: typing-extensions in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pytorch-lightning==1.4.0) (3.7.4.3)
    Requirement already satisfied: numpy>=1.17.2 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pytorch-lightning==1.4.0) (1.19.5)
    Requirement already satisfied: fsspec[http]!=2021.06.0,>=2021.05.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pytorch-lightning==1.4.0) (2021.7.0)
    Requirement already satisfied: PyYAML>=5.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pytorch-lightning==1.4.0) (5.4.1)
    Requirement already satisfied: packaging>=17.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pytorch-lightning==1.4.0) (21.0)
    Requirement already satisfied: pyDeprecate==0.3.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pytorch-lightning==1.4.0) (0.3.1)
    Requirement already satisfied: torchmetrics>=0.4.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pytorch-lightning==1.4.0) (0.5.0)
    Requirement already satisfied: future>=0.17.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pytorch-lightning==1.4.0) (0.18.2)
    Requirement already satisfied: tensorboard!=2.5.0,>=2.2.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pytorch-lightning==1.4.0) (2.6.0)
    Requirement already satisfied: pillow>=5.3.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from torchvision>=0.9.1) (8.3.1)
    Requirement already satisfied: requests in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (2.26.0)
    Requirement already satisfied: aiohttp in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (3.7.4.post0)
    Requirement already satisfied: pyparsing>=2.0.2 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from packaging>=17.0->pytorch-lightning==1.4.0) (2.4.7)
    Requirement already satisfied: absl-py>=0.4 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (0.13.0)
    Requirement already satisfied: setuptools>=41.0.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (52.0.0.post20210125)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (0.6.1)
    Requirement already satisfied: grpcio>=1.24.3 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (1.32.0)
    Requirement already satisfied: wheel>=0.26 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (0.37.0)
    Requirement already satisfied: protobuf>=3.6.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (3.17.3)
    Requirement already satisfied: google-auth<2,>=1.6.3 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (1.35.0)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (0.4.5)
    Requirement already satisfied: markdown>=2.6.8 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (3.3.4)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (1.8.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (2.0.1)
    Requirement already satisfied: six in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from absl-py>=0.4->tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (1.15.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from google-auth<2,>=1.6.3->tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (4.7.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from google-auth<2,>=1.6.3->tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (0.2.8)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from google-auth<2,>=1.6.3->tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (4.2.2)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (1.3.0)
    Requirement already satisfied: importlib-metadata in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from markdown>=2.6.8->tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (4.6.4)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (0.4.8)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (1.26.6)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (2.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (2021.5.30)
    Requirement already satisfied: idna<4,>=2.5 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (3.2)
    Requirement already satisfied: oauthlib>=3.0.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (3.1.1)
    Requirement already satisfied: databricks-cli>=0.8.7 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (0.15.0)
    Requirement already satisfied: Flask in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (2.0.1)
    Requirement already satisfied: alembic<=1.4.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (1.4.1)
    Requirement already satisfied: pytz in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (2021.1)
    Requirement already satisfied: pandas in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (1.0.1)
    Requirement already satisfied: sqlalchemy in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (1.4.23)
    Requirement already satisfied: querystring-parser in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (1.2.4)
    Requirement already satisfied: gunicorn in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (20.1.0)
    Requirement already satisfied: prometheus-flask-exporter in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (0.18.2)
    Requirement already satisfied: gitpython>=2.1.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (3.1.18)
    Requirement already satisfied: cloudpickle in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (1.6.0)
    Requirement already satisfied: docker>=4.0.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (5.0.0)
    Requirement already satisfied: entrypoints in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (0.3)
    Requirement already satisfied: sqlparse>=0.3.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (0.4.1)
    Requirement already satisfied: click>=7.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from mlflow) (7.1.2)
    Requirement already satisfied: python-editor>=0.3 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from alembic<=1.4.1->mlflow) (1.0.4)
    Requirement already satisfied: python-dateutil in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from alembic<=1.4.1->mlflow) (2.8.2)
    Requirement already satisfied: Mako in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from alembic<=1.4.1->mlflow) (1.1.5)
    Requirement already satisfied: tabulate>=0.7.7 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from databricks-cli>=0.8.7->mlflow) (0.8.9)
    Requirement already satisfied: websocket-client>=0.32.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from docker>=4.0.0->mlflow) (1.2.1)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from gitpython>=2.1.0->mlflow) (4.0.7)
    Requirement already satisfied: smmap<5,>=3.0.1 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython>=2.1.0->mlflow) (4.0.0)
    Requirement already satisfied: zipp>=0.5 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from importlib-metadata->markdown>=2.6.8->tensorboard!=2.5.0,>=2.2.0->pytorch-lightning==1.4.0) (3.5.0)
    Requirement already satisfied: greenlet!=0.4.17 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from sqlalchemy->mlflow) (1.1.1)
    Requirement already satisfied: yarl<2.0,>=1.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (1.6.3)
    Requirement already satisfied: multidict<7.0,>=4.5 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (5.1.0)
    Requirement already satisfied: async-timeout<4.0,>=3.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (3.0.1)
    Requirement already satisfied: chardet<5.0,>=2.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (4.0.0)
    Requirement already satisfied: attrs>=17.3.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning==1.4.0) (20.3.0)
    Requirement already satisfied: Jinja2>=3.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from Flask->mlflow) (3.0.1)
    Requirement already satisfied: itsdangerous>=2.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from Flask->mlflow) (2.0.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from Jinja2>=3.0->Flask->mlflow) (2.0.1)
    Requirement already satisfied: prometheus-client in /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages (from prometheus-flask-exporter->mlflow) (0.11.0)



```python
!rm -fr /tmp/mlflow
```


```python
%cd /tmp
```

    /tmp



```python
!git clone https://github.com/mlflow/mlflow.git
```

    Cloning into 'mlflow'...
    remote: Enumerating objects: 43208, done.[K
    remote: Counting objects: 100% (1650/1650), done.[K
    remote: Compressing objects: 100% (857/857), done.[K
    remote: Total 43208 (delta 992), reused 1271 (delta 767), pack-reused 41558[K
    Receiving objects: 100% (43208/43208), 38.22 MiB | 8.91 MiB/s, done.
    Resolving deltas: 100% (33107/33107), done.


### Train model using `mlflow`


```python
%cd mlflow/examples/pytorch/MNIST
```

    /tmp/mlflow/examples/pytorch/MNIST



```python
!mlflow run . --no-conda
```

    2021/09/07 09:54:39 INFO mlflow.projects.utils: === Created directory /tmp/tmp583_9vom for downloading remote URIs passed to arguments of type 'path' ===
    2021/09/07 09:54:39 INFO mlflow.projects.backend.local: === Running command 'python mnist_autolog_example.py \
      --max_epochs 5 \
      --gpus 0 \
      --accelerator None \
      --batch_size 64 \
      --num_workers 3 \
      --lr 0.001 \
      --es_patience 3 \
      --es_mode min \
      --es_verbose True \
      --es_monitor val_loss
    ' in run with ID 'c478ef52c6ec4c2fbaba3ad9f788a101' === 
    /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
    /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:446: UserWarning: Checkpoint directory /tmp/mlflow/examples/pytorch/MNIST exists and is not empty.
      rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
    GPU available: True, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/pytorch_lightning/trainer/trainer.py:1293: UserWarning: GPU available but not used. Set the gpus flag in your trainer `Trainer(gpus=1)` or script `--gpus=1`.
      "GPU available but not used. Set the gpus flag in your trainer"
    2021/09/07 09:54:40 INFO mlflow.utils.autologging_utils: Created MLflow autologging run with ID 'c478ef52c6ec4c2fbaba3ad9f788a101', which will track hyperparameters, performance metrics, model artifacts, and lineage information for the current pytorch workflow
    /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/pytorch_lightning/core/datamodule.py:424: LightningDeprecationWarning: DataModule.setup has already been called, so it will not be called again. In v1.6 this behavior will change to always call DataModule.setup.
      f"DataModule.{name} has already been called, so it will not be called again. "
    2021-09-07 09:54:41.067061: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2021-09-07 09:54:41.067084: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    
      | Name    | Type   | Params
    -----------------------------------
    0 | layer_1 | Linear | 100 K 
    1 | layer_2 | Linear | 33.0 K
    2 | layer_3 | Linear | 2.6 K 
    -----------------------------------
    136 K     Trainable params
    0         Non-trainable params
    136 K     Total params
    0.544     Total estimated model params size (MB)
    Epoch 0:  92%|█████████▏| 860/939 [00:04<00:00, 176.21it/s, loss=0.157, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 0:  92%|█████████▏| 866/939 [00:04<00:00, 175.08it/s, loss=0.157, v_num=1][A
    Epoch 0:  94%|█████████▍| 887/939 [00:05<00:00, 175.52it/s, loss=0.157, v_num=1][A
    Epoch 0:  97%|█████████▋| 910/939 [00:05<00:00, 176.58it/s, loss=0.157, v_num=1][A
    Epoch 0:  99%|█████████▉| 933/939 [00:05<00:00, 176.97it/s, loss=0.157, v_num=1][A
    Epoch 0: 100%|██████████| 939/939 [00:05<00:00, 176.38it/s, loss=0.157, v_num=1][A
                                                                                    [AMetric val_loss improved. New best score: 0.139
    Epoch 0, global step 859: val_loss reached 0.13937 (best 0.13937), saving model to "/tmp/mlflow/examples/pytorch/MNIST/epoch=0-step=859.ckpt" as top 1
    /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/pytorch_lightning/trainer/callback_hook.py:103: LightningDeprecationWarning: The signature of `Callback.on_train_epoch_end` has changed in v1.3. `outputs` parameter has been removed. Support for the old signature will be removed in v1.5
      "The signature of `Callback.on_train_epoch_end` has changed in v1.3."
    /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/pytorch_lightning/trainer/deprecated_api.py:26: LightningDeprecationWarning: `Trainer.running_sanity_check` has been renamed to `Trainer.sanity_checking` and will be removed in v1.5.
      "`Trainer.running_sanity_check` has been renamed to `Trainer.sanity_checking` and will be removed in v1.5."
    Epoch 1:  92%|████████▏| 860/939 [00:04<00:00, 174.51it/s, loss=0.0984, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Validating:   0%|                                        | 0/79 [00:00<?, ?it/s][A
    Epoch 1:  93%|████████▍| 874/939 [00:05<00:00, 172.87it/s, loss=0.0984, v_num=1][A
    Epoch 1:  96%|████████▌| 897/939 [00:05<00:00, 173.75it/s, loss=0.0984, v_num=1][A
    Epoch 1:  98%|████████▊| 920/939 [00:05<00:00, 174.23it/s, loss=0.0984, v_num=1][A
    Epoch 1: 100%|█████████| 939/939 [00:05<00:00, 174.44it/s, loss=0.0984, v_num=1][A
                                                                                    [AMetric val_loss improved by 0.028 >= min_delta = 0.0. New best score: 0.112
    Epoch 1, global step 1719: val_loss reached 0.11164 (best 0.11164), saving model to "/tmp/mlflow/examples/pytorch/MNIST/epoch=1-step=1719-v1.ckpt" as top 1
    Epoch 2:  92%|████████▏| 860/939 [00:05<00:00, 170.63it/s, loss=0.0723, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Validating:   0%|                                        | 0/79 [00:00<?, ?it/s][A
    Epoch 2:  93%|████████▍| 874/939 [00:05<00:00, 168.68it/s, loss=0.0723, v_num=1][A
    Epoch 2:  96%|████████▌| 897/939 [00:05<00:00, 169.39it/s, loss=0.0723, v_num=1][A
    Epoch 2:  98%|████████▊| 920/939 [00:05<00:00, 170.16it/s, loss=0.0723, v_num=1][A
    Epoch 2: 100%|█████████| 939/939 [00:05<00:00, 170.58it/s, loss=0.0723, v_num=1][A
                                                                                    [AEpoch 2, global step 2579: val_loss was not in top 1
    Epoch 3:  92%|████████▏| 860/939 [00:04<00:00, 181.61it/s, loss=0.0594, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Validating:   0%|                                        | 0/79 [00:00<?, ?it/s][A
    Epoch 3:  93%|████████▍| 874/939 [00:04<00:00, 179.31it/s, loss=0.0594, v_num=1][A
    Epoch 3:  96%|████████▌| 897/939 [00:05<00:00, 179.57it/s, loss=0.0594, v_num=1][A
    Epoch 3:  98%|████████▊| 920/939 [00:05<00:00, 180.30it/s, loss=0.0594, v_num=1][A
    Epoch 3: 100%|█████████| 939/939 [00:05<00:00, 180.52it/s, loss=0.0594, v_num=1][A
                                                                                    [AMetric val_loss improved by 0.004 >= min_delta = 0.0. New best score: 0.108
    Epoch 3, global step 3439: val_loss reached 0.10778 (best 0.10778), saving model to "/tmp/mlflow/examples/pytorch/MNIST/epoch=3-step=3439.ckpt" as top 1
    Epoch 4:  92%|████████▏| 860/939 [00:04<00:00, 174.71it/s, loss=0.0495, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Validating:   0%|                                        | 0/79 [00:00<?, ?it/s][A
    Epoch 4:  93%|████████▍| 874/939 [00:05<00:00, 173.36it/s, loss=0.0495, v_num=1][A
    Epoch 4:  96%|████████▌| 897/939 [00:05<00:00, 173.43it/s, loss=0.0495, v_num=1][A
    Epoch 4:  98%|████████▊| 920/939 [00:05<00:00, 174.46it/s, loss=0.0495, v_num=1][A
    Epoch 4: 100%|█████████| 939/939 [00:05<00:00, 174.89it/s, loss=0.0495, v_num=1][A
                                                                                    [AEpoch 4, global step 4299: val_loss was not in top 1
    Epoch 4: 100%|█████████| 939/939 [00:05<00:00, 174.80it/s, loss=0.0495, v_num=1]
    2021/09/07 09:55:09 WARNING mlflow.utils.autologging_utils: MLflow autologging encountered a warning: "/home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/pytorch_lightning/core/memory.py:203: LightningDeprecationWarning: Argument `mode` in `ModelSummary` is deprecated in v1.4 and will be removed in v1.6. Use `max_depth=-1` to replicate `mode=full` behaviour."
    /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/pytorch_lightning/core/datamodule.py:424: LightningDeprecationWarning: DataModule.prepare_data has already been called, so it will not be called again. In v1.6 this behavior will change to always call DataModule.prepare_data.
      f"DataModule.{name} has already been called, so it will not be called again. "
    Testing: 0it [00:00, ?it/s]/home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/deprecate/deprecation.py:115: LightningDeprecationWarning: The `accuracy` was deprecated since v1.3.0 in favor of `torchmetrics.functional.classification.accuracy.accuracy`. It will be removed in v1.5.0.
      stream(template_mgs % msg_args)
    Testing:  91%|████████████████████████████▏  | 143/157 [00:00<00:00, 205.75it/s]--------------------------------------------------------------------------------
    DATALOADER:0 TEST RESULTS
    {'avg_test_acc': 0.9678543210029602}
    --------------------------------------------------------------------------------
    Testing: 100%|███████████████████████████████| 157/157 [00:00<00:00, 189.99it/s]
    2021/09/07 09:55:12 INFO mlflow.projects: === Run (ID 'c478ef52c6ec4c2fbaba3ad9f788a101') succeeded ===



```python
!tree -L 1 mlruns/0
```

    [01;34mmlruns/0[00m
    ├── [01;34m3243d1faac7c4eb69ee11e2b77766e2a[00m
    ├── [01;34mc478ef52c6ec4c2fbaba3ad9f788a101[00m
    └── meta.yaml
    
    2 directories, 1 file


### Choose test image


```python
from torchvision import datasets

mnist_test = datasets.MNIST('/tmp/data', train=False, download=True)
# change the index below to get a different image for testing
mnist_test = list(mnist_test)[0]
img, category = mnist_test
display(img)
print(category)
```

    /home/sa/miniconda3/envs/tempo-examples/lib/python3.7/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)



    
![png](README_files/README_13_1.png)
    


    7


### Tranform test image to numpy


```python
import numpy as np
img_np = np.asarray(img).reshape((1, 28*28)).astype(np.float32)
```

## Save model environment


```python
import glob
import os

files = glob.glob("mlruns/0/*/")
files.sort(key=os.path.getmtime)

ARTIFACTS_FOLDER = os.path.join(
    os.getcwd(),
    files[-1],
    "artifacts",
    "model"
)
assert os.path.exists(ARTIFACTS_FOLDER)
print(ARTIFACTS_FOLDER)
```

    /tmp/mlflow/examples/pytorch/MNIST/mlruns/0/c478ef52c6ec4c2fbaba3ad9f788a101/artifacts/model


### Define `tempo` model


```python
from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model

mlflow_tag = "mlflow"

pytorch_mnist_model = Model(
    name="test-pytorch-mnist",
    platform=ModelFramework.MLFlow,
    local_folder=ARTIFACTS_FOLDER,
    # if we deploy to kube, this defines where the model artifacts are stored
    uri="s3://tempo/basic/mnist",
    description="A pytorch MNIST model",
)


```

    Insights Manager not initialised as empty URL provided.


### Save model (environment) using `tempo`

Tempo hides many details required to save the model environment for `mlserver`:
- Add required runtime dependencies
- Create a conda pack `environment.tar.gz`


```python
from tempo.serve.loader import save
save(pytorch_mnist_model)
```

    Collecting package metadata (repodata.json): ...working... done
    Solving environment: ...working... done
    
    Downloading and Extracting Packages
    _openmp_mutex-4.5    | 22 KB     | ########## | 100% 
    Preparing transaction: ...working... done
    Verifying transaction: ...working... done
    Executing transaction: ...working... done
    Installing pip dependencies: ...working... Ran pip subprocess with arguments:
    ['/home/sa/miniconda3/envs/tempo-74889c69-e5a7-499d-9d10-867b916d697c/bin/python', '-m', 'pip', 'install', '-U', '-r', '/tmp/condaenv.hfliozx3.requirements.txt']
    Pip subprocess output:
    Collecting mlflow
      Using cached mlflow-1.20.2-py3-none-any.whl (14.6 MB)
    Collecting astunparse==1.6.3
      Using cached astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Collecting cffi==1.14.6
      Using cached cffi-1.14.6-cp37-cp37m-manylinux1_x86_64.whl (402 kB)
    Collecting cloudpickle==1.6.0
      Using cached cloudpickle-1.6.0-py3-none-any.whl (23 kB)
    Collecting defusedxml==0.7.1
      Using cached defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)
    Collecting dill==0.3.4
      Using cached dill-0.3.4-py2.py3-none-any.whl (86 kB)
    Collecting ipython==7.26.0
      Using cached ipython-7.26.0-py3-none-any.whl (786 kB)
    Collecting pydeprecate==0.3.1
      Using cached pyDeprecate-0.3.1-py3-none-any.whl (10 kB)
    Collecting pytorch-lightning==1.4.0
      Using cached pytorch_lightning-1.4.0-py3-none-any.whl (913 kB)
    Collecting scipy==1.7.1
      Using cached scipy-1.7.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (28.5 MB)
    Collecting torch==1.9.0
      Using cached torch-1.9.0-cp37-cp37m-manylinux1_x86_64.whl (831.4 MB)
    Collecting torchvision==0.10.0
      Using cached torchvision-0.10.0-cp37-cp37m-manylinux1_x86_64.whl (22.1 MB)
    Collecting mlserver==0.4.1.dev1
      Using cached mlserver-0.4.1.dev1-py3-none-any.whl (63 kB)
    Collecting mlserver-mlflow==0.4.1.dev1
      Using cached mlserver_mlflow-0.4.1.dev1-py3-none-any.whl (8.6 kB)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /home/sa/miniconda3/envs/tempo-74889c69-e5a7-499d-9d10-867b916d697c/lib/python3.7/site-packages (from astunparse==1.6.3->-r /tmp/condaenv.hfliozx3.requirements.txt (line 2)) (0.37.0)
    Collecting six<2.0,>=1.6.1
      Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
    Collecting pycparser
      Using cached pycparser-2.20-py2.py3-none-any.whl (112 kB)
    Requirement already satisfied: setuptools>=18.5 in /home/sa/miniconda3/envs/tempo-74889c69-e5a7-499d-9d10-867b916d697c/lib/python3.7/site-packages (from ipython==7.26.0->-r /tmp/condaenv.hfliozx3.requirements.txt (line 7)) (57.4.0)
    Collecting pickleshare
      Using cached pickleshare-0.7.5-py2.py3-none-any.whl (6.9 kB)
    Collecting traitlets>=4.2
      Using cached traitlets-5.1.0-py3-none-any.whl (101 kB)
    Collecting decorator
      Using cached decorator-5.0.9-py3-none-any.whl (8.9 kB)
    Collecting jedi>=0.16
      Using cached jedi-0.18.0-py2.py3-none-any.whl (1.4 MB)
    Collecting pygments
      Using cached Pygments-2.10.0-py3-none-any.whl (1.0 MB)
    Collecting pexpect>4.3
      Using cached pexpect-4.8.0-py2.py3-none-any.whl (59 kB)
    Collecting matplotlib-inline
      Using cached matplotlib_inline-0.1.2-py3-none-any.whl (8.2 kB)
    Collecting prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0
      Using cached prompt_toolkit-3.0.20-py3-none-any.whl (370 kB)
    Collecting backcall
      Using cached backcall-0.2.0-py2.py3-none-any.whl (11 kB)
    Collecting tqdm>=4.41.0
      Using cached tqdm-4.62.2-py2.py3-none-any.whl (76 kB)
    Collecting tensorboard!=2.5.0,>=2.2.0
      Using cached tensorboard-2.6.0-py3-none-any.whl (5.6 MB)
    Collecting torchmetrics>=0.4.0
      Using cached torchmetrics-0.5.1-py3-none-any.whl (282 kB)
    Collecting packaging>=17.0
      Using cached packaging-21.0-py3-none-any.whl (40 kB)
    Collecting numpy>=1.17.2
      Using cached numpy-1.21.2-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.7 MB)
    Collecting PyYAML>=5.1
      Using cached PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)
    Collecting typing-extensions
      Using cached typing_extensions-3.10.0.2-py3-none-any.whl (26 kB)
    Collecting fsspec[http]!=2021.06.0,>=2021.05.0
      Using cached fsspec-2021.8.1-py3-none-any.whl (119 kB)
    Collecting future>=0.17.1
      Using cached future-0.18.2-py3-none-any.whl
    Collecting pillow>=5.3.0
      Using cached Pillow-8.3.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.0 MB)
    Collecting pandas
      Using cached pandas-1.3.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.3 MB)
    Collecting uvicorn
      Using cached uvicorn-0.15.0-py3-none-any.whl (54 kB)
    Collecting protobuf
      Using cached protobuf-3.17.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)
    Collecting click
      Using cached click-8.0.1-py3-none-any.whl (97 kB)
    Collecting fastapi
      Using cached fastapi-0.68.1-py3-none-any.whl (52 kB)
    Collecting orjson
      Using cached orjson-3.6.3-cp37-cp37m-manylinux_2_24_x86_64.whl (234 kB)
    Collecting grpcio
      Using cached grpcio-1.39.0-cp37-cp37m-manylinux2014_x86_64.whl (4.3 MB)
    Collecting gunicorn
      Using cached gunicorn-20.1.0-py3-none-any.whl (79 kB)
    Collecting databricks-cli>=0.8.7
      Using cached databricks_cli-0.15.0-py3-none-any.whl
    Collecting alembic<=1.4.1
      Using cached alembic-1.4.1-py2.py3-none-any.whl
    Collecting entrypoints
      Using cached entrypoints-0.3-py2.py3-none-any.whl (11 kB)
    Collecting Flask
      Using cached Flask-2.0.1-py3-none-any.whl (94 kB)
    Collecting requests>=2.17.3
      Using cached requests-2.26.0-py2.py3-none-any.whl (62 kB)
    Collecting prometheus-flask-exporter
      Using cached prometheus_flask_exporter-0.18.2-py3-none-any.whl
    Collecting querystring-parser
      Using cached querystring_parser-1.2.4-py2.py3-none-any.whl (7.9 kB)
    Collecting sqlparse>=0.3.1
      Using cached sqlparse-0.4.1-py3-none-any.whl (42 kB)
    Collecting gitpython>=2.1.0
      Using cached GitPython-3.1.18-py3-none-any.whl (170 kB)
    Collecting sqlalchemy
      Using cached SQLAlchemy-1.4.23-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.5 MB)
    Collecting importlib-metadata!=4.7.0,>=3.7.0
      Using cached importlib_metadata-4.8.1-py3-none-any.whl (17 kB)
    Collecting docker>=4.0.0
      Using cached docker-5.0.2-py2.py3-none-any.whl (145 kB)
    Collecting pytz
      Using cached pytz-2021.1-py2.py3-none-any.whl (510 kB)
    Collecting python-editor>=0.3
      Using cached python_editor-1.0.4-py3-none-any.whl (4.9 kB)
    Collecting Mako
      Using cached Mako-1.1.5-py2.py3-none-any.whl (75 kB)
    Collecting python-dateutil
      Using cached python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)
    Collecting tabulate>=0.7.7
      Using cached tabulate-0.8.9-py3-none-any.whl (25 kB)
    Collecting websocket-client>=0.32.0
      Using cached websocket_client-1.2.1-py2.py3-none-any.whl (52 kB)
    Collecting aiohttp
      Using cached aiohttp-3.7.4.post0-cp37-cp37m-manylinux2014_x86_64.whl (1.3 MB)
    Collecting gitdb<5,>=4.0.1
      Using cached gitdb-4.0.7-py3-none-any.whl (63 kB)
    Collecting smmap<5,>=3.0.1
      Using cached smmap-4.0.0-py2.py3-none-any.whl (24 kB)
    Collecting zipp>=0.5
      Using cached zipp-3.5.0-py3-none-any.whl (5.7 kB)
    Collecting parso<0.9.0,>=0.8.0
      Using cached parso-0.8.2-py2.py3-none-any.whl (94 kB)
    Collecting pyparsing>=2.0.2
      Using cached pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)
    Collecting ptyprocess>=0.5
      Using cached ptyprocess-0.7.0-py2.py3-none-any.whl (13 kB)
    Collecting wcwidth
      Using cached wcwidth-0.2.5-py2.py3-none-any.whl (30 kB)
    Collecting certifi>=2017.4.17
      Using cached certifi-2021.5.30-py2.py3-none-any.whl (145 kB)
    Collecting charset-normalizer~=2.0.0
      Using cached charset_normalizer-2.0.4-py3-none-any.whl (36 kB)
    Collecting idna<4,>=2.5
      Using cached idna-3.2-py3-none-any.whl (59 kB)
    Collecting urllib3<1.27,>=1.21.1
      Using cached urllib3-1.26.6-py2.py3-none-any.whl (138 kB)
    Collecting greenlet!=0.4.17
      Using cached greenlet-1.1.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (143 kB)
    Collecting google-auth<2,>=1.6.3
      Using cached google_auth-1.35.0-py2.py3-none-any.whl (152 kB)
    Collecting markdown>=2.6.8
      Using cached Markdown-3.3.4-py3-none-any.whl (97 kB)
    Collecting google-auth-oauthlib<0.5,>=0.4.1
      Using cached google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Collecting absl-py>=0.4
      Using cached absl_py-0.13.0-py3-none-any.whl (132 kB)
    Collecting tensorboard-plugin-wit>=1.6.0
      Using cached tensorboard_plugin_wit-1.8.0-py3-none-any.whl (781 kB)
    Collecting tensorboard-data-server<0.7.0,>=0.6.0
      Using cached tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl (4.9 MB)
    Collecting werkzeug>=0.11.15
      Using cached Werkzeug-2.0.1-py3-none-any.whl (288 kB)
    Collecting rsa<5,>=3.1.4
      Using cached rsa-4.7.2-py3-none-any.whl (34 kB)
    Collecting cachetools<5.0,>=2.0.0
      Using cached cachetools-4.2.2-py3-none-any.whl (11 kB)
    Collecting pyasn1-modules>=0.2.1
      Using cached pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)
    Collecting requests-oauthlib>=0.7.0
      Using cached requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)
    Collecting pyasn1<0.5.0,>=0.4.6
      Using cached pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)
    Collecting oauthlib>=3.0.0
      Using cached oauthlib-3.1.1-py2.py3-none-any.whl (146 kB)
    Collecting async-timeout<4.0,>=3.0
      Using cached async_timeout-3.0.1-py3-none-any.whl (8.2 kB)
    Collecting yarl<2.0,>=1.0
      Using cached yarl-1.6.3-cp37-cp37m-manylinux2014_x86_64.whl (294 kB)
    Collecting multidict<7.0,>=4.5
      Using cached multidict-5.1.0-cp37-cp37m-manylinux2014_x86_64.whl (142 kB)
    Collecting chardet<5.0,>=2.0
      Using cached chardet-4.0.0-py2.py3-none-any.whl (178 kB)
    Collecting attrs>=17.3.0
      Using cached attrs-21.2.0-py2.py3-none-any.whl (53 kB)
    Collecting starlette==0.14.2
      Using cached starlette-0.14.2-py3-none-any.whl (60 kB)
    Collecting pydantic!=1.7,!=1.7.1,!=1.7.2,!=1.7.3,!=1.8,!=1.8.1,<2.0.0,>=1.6.2
      Using cached pydantic-1.8.2-cp37-cp37m-manylinux2014_x86_64.whl (10.1 MB)
    Collecting itsdangerous>=2.0
      Using cached itsdangerous-2.0.1-py3-none-any.whl (18 kB)
    Collecting Jinja2>=3.0
      Using cached Jinja2-3.0.1-py3-none-any.whl (133 kB)
    Collecting MarkupSafe>=2.0
      Using cached MarkupSafe-2.0.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (31 kB)
    Collecting prometheus-client
      Using cached prometheus_client-0.11.0-py2.py3-none-any.whl (56 kB)
    Collecting asgiref>=3.4.0
      Using cached asgiref-3.4.1-py3-none-any.whl (25 kB)
    Collecting h11>=0.8
      Using cached h11-0.12.0-py3-none-any.whl (54 kB)
    Installing collected packages: zipp, typing-extensions, urllib3, pyasn1, MarkupSafe, importlib-metadata, idna, charset-normalizer, certifi, werkzeug, smmap, six, rsa, requests, pyasn1-modules, oauthlib, multidict, Jinja2, itsdangerous, greenlet, click, cachetools, yarl, websocket-client, tabulate, starlette, sqlalchemy, requests-oauthlib, pytz, python-editor, python-dateutil, pyparsing, pydantic, prometheus-client, numpy, Mako, h11, google-auth, gitdb, Flask, chardet, attrs, async-timeout, asgiref, wcwidth, uvicorn, traitlets, torch, tensorboard-plugin-wit, tensorboard-data-server, sqlparse, querystring-parser, PyYAML, ptyprocess, protobuf, prometheus-flask-exporter, parso, pandas, packaging, orjson, markdown, gunicorn, grpcio, google-auth-oauthlib, gitpython, fsspec, fastapi, entrypoints, docker, databricks-cli, cloudpickle, alembic, aiohttp, absl-py, tqdm, torchmetrics, tensorboard, pygments, pydeprecate, pycparser, prompt-toolkit, pillow, pickleshare, pexpect, mlserver, mlflow, matplotlib-inline, jedi, future, decorator, backcall, torchvision, scipy, pytorch-lightning, mlserver-mlflow, ipython, dill, defusedxml, cffi, astunparse
    Successfully installed Flask-2.0.1 Jinja2-3.0.1 Mako-1.1.5 MarkupSafe-2.0.1 PyYAML-5.4.1 absl-py-0.13.0 aiohttp-3.7.4.post0 alembic-1.4.1 asgiref-3.4.1 astunparse-1.6.3 async-timeout-3.0.1 attrs-21.2.0 backcall-0.2.0 cachetools-4.2.2 certifi-2021.5.30 cffi-1.14.6 chardet-4.0.0 charset-normalizer-2.0.4 click-8.0.1 cloudpickle-1.6.0 databricks-cli-0.15.0 decorator-5.0.9 defusedxml-0.7.1 dill-0.3.4 docker-5.0.2 entrypoints-0.3 fastapi-0.68.1 fsspec-2021.8.1 future-0.18.2 gitdb-4.0.7 gitpython-3.1.18 google-auth-1.35.0 google-auth-oauthlib-0.4.6 greenlet-1.1.1 grpcio-1.39.0 gunicorn-20.1.0 h11-0.12.0 idna-3.2 importlib-metadata-4.8.1 ipython-7.26.0 itsdangerous-2.0.1 jedi-0.18.0 markdown-3.3.4 matplotlib-inline-0.1.2 mlflow-1.20.2 mlserver-0.4.1.dev1 mlserver-mlflow-0.4.1.dev1 multidict-5.1.0 numpy-1.21.2 oauthlib-3.1.1 orjson-3.6.3 packaging-21.0 pandas-1.3.2 parso-0.8.2 pexpect-4.8.0 pickleshare-0.7.5 pillow-8.3.2 prometheus-client-0.11.0 prometheus-flask-exporter-0.18.2 prompt-toolkit-3.0.20 protobuf-3.17.3 ptyprocess-0.7.0 pyasn1-0.4.8 pyasn1-modules-0.2.8 pycparser-2.20 pydantic-1.8.2 pydeprecate-0.3.1 pygments-2.10.0 pyparsing-2.4.7 python-dateutil-2.8.2 python-editor-1.0.4 pytorch-lightning-1.4.0 pytz-2021.1 querystring-parser-1.2.4 requests-2.26.0 requests-oauthlib-1.3.0 rsa-4.7.2 scipy-1.7.1 six-1.16.0 smmap-4.0.0 sqlalchemy-1.4.23 sqlparse-0.4.1 starlette-0.14.2 tabulate-0.8.9 tensorboard-2.6.0 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.0 torch-1.9.0 torchmetrics-0.5.1 torchvision-0.10.0 tqdm-4.62.2 traitlets-5.1.0 typing-extensions-3.10.0.2 urllib3-1.26.6 uvicorn-0.15.0 wcwidth-0.2.5 websocket-client-1.2.1 werkzeug-2.0.1 yarl-1.6.3 zipp-3.5.0
    
    done
    #
    # To activate this environment, use
    #
    #     $ conda activate tempo-74889c69-e5a7-499d-9d10-867b916d697c
    #
    # To deactivate an active environment, use
    #
    #     $ conda deactivate
    
    Collecting packages...
    Packing environment at '/home/sa/miniconda3/envs/tempo-74889c69-e5a7-499d-9d10-867b916d697c' to '/tmp/mlflow/examples/pytorch/MNIST/mlruns/0/c478ef52c6ec4c2fbaba3ad9f788a101/artifacts/model/environment.tar.gz'
    [#############################           ] | 73% Completed | 58.6s

## Deploy to Docker


```python
from tempo import deploy_local
local_deployed_model = deploy_local(pytorch_mnist_model)
```


```python
local_prediction = local_deployed_model.predict(img_np)
print(np.nonzero(local_prediction.flatten() == 0))
```


```python
local_deployed_model.undeploy()
```

## Deploy to Kubernetes

### Prerequisites
 
Create a Kind Kubernetes cluster with Minio and Seldon Core installed using Ansible as described [here](https://tempo.readthedocs.io/en/latest/overview/quickstart.html#kubernetes-cluster-with-seldon-core).


```python
%cd -0
```


```python
!kubectl apply -f k8s/rbac -n seldon
```

### Upload artifacts to minio


```python
from tempo.examples.minio import create_minio_rclone
import os
create_minio_rclone(os.getcwd()+"/rclone.conf")
```


```python
from tempo.serve.loader import upload
upload(pytorch_mnist_model)
```


```python
from tempo.serve.metadata import SeldonCoreOptions
runtime_options = SeldonCoreOptions(**{
        "remote_options": {
            "namespace": "seldon",
            "authSecretName": "minio-secret"
        }
    })
```

### Deploy to `kind`


```python
from tempo import deploy_remote
remote_deployed_model = deploy_remote(pytorch_mnist_model, options=runtime_options)
```


```python
remote_prediction = remote_deployed_model.predict(img_np)
print(np.nonzero(remote_prediction.flatten() == 0))
```


```python
remote_deployed_model.undeploy()
```


```python

```
