# Metaflow with Tempo Example

We will train two models and deploy them with tempo within a Metaflow pipeline. To understand the core example see [here](https://tempo.readthedocs.io/en/latest/examples/multi-model/README.html)

![archtecture](architecture.png)

## MetaFlow Prequisites


### Install metaflow locally

```
pip install metaflow
```

### Setup AWS Metaflow Support

Note at present this is required even for a local run as artifacts are stored on S3.

[Install Metaflow with remote AWS support](https://docs.metaflow.org/metaflow-on-aws/metaflow-on-aws).

### Setup Conda-Forge Support

The flow will use conda-forge so you need to add that channel to conda.

```
conda config --add channels conda-forge
```



## Tempo Requirements

For deploying to a remote Kubernetes cluster with Seldon Core installed do the following steps:

### GKE Cluster

Create a GKE cluster and install Seldon Core on it using [Ansible to install Seldon Core on a Kubernetes cluster](https://github.com/SeldonIO/ansible-k8s-collection).

For Metaflow to connect to your GKE cluster from AWS it will need to authenticate inside the running pipeline. For this you will need to do some further steps as follows:

You will need to create two files in the flow src folder:

```bash
kubeconfig.yaml
gsa-key.json
```

Follow the steps outlined in [GKE server authentication](https://cloud.google.com/kubernetes-engine/docs/how-to/api-server-authentication#environments-without-gcloud).



## Iris Flow Summary


```python
!python src/irisflow.py --environment=conda show
```

    [35m[1mMetaflow 2.3.2[0m[35m[22m executing [0m[31m[1mIrisFlow[0m[35m[22m[0m[35m[22m for [0m[31m[1muser:clive[0m[35m[22m[K[0m[35m[22m[0m
    [22m
    A Flow to train two Iris dataset models and combine them for inference with Tempo
    
    The flow performs the following steps:
    
    1) Load Iris Data
    2) Train SKLearn LR Model
    3) Train XGBoost LR Model
    4) Create and deploy Tempo artifacts[K[0m[22m[0m
    [22m
    Step [0m[31m[1mstart[0m[22m[K[0m[22m[0m
    [22m    Download Iris classification datatset[K[0m[22m[0m
    [22m    [0m[35m[22m=>[0m[22m [0m[35m[22mtrain_sklearn[0m[22m, [0m[35m[22mtrain_xgboost[0m[22m[K[0m[22m[0m
    [22m
    Step [0m[31m[1mtrain_sklearn[0m[22m[K[0m[22m[0m
    [22m    Train a SKLearn Logistic Regression Classifier on dataset and save model as artifact[K[0m[22m[0m
    [22m    [0m[35m[22m=>[0m[22m [0m[35m[22mjoin[0m[22m[K[0m[22m[0m
    [22m
    Step [0m[31m[1mtrain_xgboost[0m[22m[K[0m[22m[0m
    [22m    Train an XGBoost classifier on the dataset and save model as artifact[K[0m[22m[0m
    [22m    [0m[35m[22m=>[0m[22m [0m[35m[22mjoin[0m[22m[K[0m[22m[0m
    [22m
    Step [0m[31m[1mjoin[0m[22m[K[0m[22m[0m
    [22m    Merge two training runs.[K[0m[22m[0m
    [22m    [0m[35m[22m=>[0m[22m [0m[35m[22mtempo[0m[22m[K[0m[22m[0m
    [22m
    Step [0m[31m[1mtempo[0m[22m[K[0m[22m[0m
    [22m    Create Tempo artifacts locally and saved to S3 within the workflow bucket.
        Then either deploy locally to Docker or deploy to a remote Kubernetes cluster based on the
        --tempo-on-docker parameter[K[0m[22m[0m
    [22m    [0m[35m[22m=>[0m[22m [0m[35m[22mend[0m[22m[K[0m[22m[0m
    [22m
    Step [0m[31m[1mend[0m[22m[K[0m[22m[0m
    [22m    End flow.[K[0m[22m[0m
    [22m[K[0m[22m[0m



```python
!python src/irisflow.py --environment=conda run --help
```

    [35m[1mMetaflow 2.3.2[0m[35m[22m executing [0m[31m[1mIrisFlow[0m[35m[22m[0m[35m[22m for [0m[31m[1muser:clive[0m[35m[22m[K[0m[35m[22m[0m
    Usage: irisflow.py run [OPTIONS]
    
      Run the workflow locally.
    
    Options:
      --conda_env FILEPATH       The path to conda environment for classifier
                                 [default: src/conda.yaml]
    
      --kubeconfig FILEPATH      The path to kubeconfig  [default:
                                 src/kubeconfig.yaml]
    
      --gsa_key FILEPATH         The path to google service account json
                                 [default: src/gsa-key.json]
    
      --tempo-on-docker BOOLEAN  Whether to deploy Tempo artifacts to Docker
                                 [default: False]
    
      --k8s_provider TEXT        kubernetes provider. Needed for non local run to
                                 deploy  [default: gke]
    
      --tag TEXT                 Annotate this run with the given tag. You can
                                 specify this option multiple times to attach
                                 multiple tags in the run.
    
      --max-workers INTEGER      Maximum number of parallel processes.  [default:
                                 16]
    
      --max-num-splits INTEGER   Maximum number of splits allowed in a foreach.
                                 This is a safety check preventing bugs from
                                 triggering thousands of steps inadvertently.
                                 [default: 100]
    
      --max-log-size INTEGER     Maximum size of stdout and stderr captured in
                                 megabytes. If a step outputs more than this to
                                 stdout/stderr, its output will be truncated.
                                 [default: 10]
    
      --with TEXT                Add a decorator to all steps. You can specify
                                 this option multiple times to attach multiple
                                 decorators in steps.
    
      --run-id-file TEXT         Write the ID of this run to the file specified.
      --namespace TEXT           Change namespace from the default (your username)
                                 to the specified tag. Note that this option does
                                 not alter tags assigned to the objects produced
                                 by this run, just what existing objects are
                                 visible in the client API. You can enable the
                                 global namespace with an empty string.--
                                 namespace=
    
      --help                     Show this message and exit.


## Run Flow locally to deploy to Docker


```python
!python src/irisflow.py \
    --environment=conda \
    run --tempo-on-docker true
```

    [35m[1mMetaflow 2.3.2[0m[35m[22m executing [0m[31m[1mIrisFlow[0m[35m[22m[0m[35m[22m for [0m[31m[1muser:clive[0m[35m[22m[K[0m[35m[22m[0m
    [35m[22mValidating your flow...[K[0m[35m[22m[0m
    [32m[1m    The graph looks good![K[0m[32m[1m[0m
    [35m[22mRunning pylint...[K[0m[35m[22m[0m
    [32m[22m    Pylint not found, so extra checks are disabled.[K[0m[32m[22m[0m
    [22mBootstrapping conda environment...(this could take a few minutes)[K[0m[22m[0m
    [22mIncluding file src/conda.yaml of size 115B [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-dq2lr7x0v0nz/metaflow/data/IrisFlow/a94f1cff7702ed70807d16917bb282f51a28511e[K[0m[22m[0m
    [22mIncluding file src/gsa-key.json of size 2KB [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-dq2lr7x0v0nz/metaflow/data/IrisFlow/6a24eaae697b14c2d8cdc2dcee1fce38b23b6ce0[K[0m[22m[0m
    [22mIncluding file src/kubeconfig.yaml of size 2KB [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-dq2lr7x0v0nz/metaflow/data/IrisFlow/00e21f5423ea21f695831e970aa81a283ac25812[K[0m[22m[0m
    [35m2021-08-08 15:29:47.141 [0m[1mWorkflow starting (run-id 31):[0m
    [35m2021-08-08 15:29:48.643 [0m[32m[31/start/165 (pid 2467618)] [0m[1mTask is starting.[0m
    [35m2021-08-08 15:29:52.492 [0m[32m[31/start/165 (pid 2467618)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:29:54.118 [0m[32m[31/train_sklearn/166 (pid 2467703)] [0m[1mTask is starting.[0m
    [35m2021-08-08 15:29:55.153 [0m[32m[31/train_xgboost/167 (pid 2467730)] [0m[1mTask is starting.[0m
    [35m2021-08-08 14:29:57.846 [0m[32m[31/train_xgboost/167 (pid 2467730)] [0m[22m[15:29:57] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.[0m
    [35m2021-08-08 15:30:00.128 [0m[32m[31/train_sklearn/166 (pid 2467703)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:30:01.230 [0m[32m[31/train_xgboost/167 (pid 2467730)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:30:03.002 [0m[32m[31/join/168 (pid 2467878)] [0m[1mTask is starting.[0m
    [35m2021-08-08 15:30:08.423 [0m[32m[31/join/168 (pid 2467878)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:30:09.919 [0m[32m[31/tempo/169 (pid 2468001)] [0m[1mTask is starting.[0m
    [35m2021-08-08 14:30:11.649 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mPip Test Install: mlops-tempo 0.4.0.dev5[0m
    [35m2021-08-08 14:30:13.798 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mPip Test Install: conda_env 2.4.2[0m
    [35m2021-08-08 14:30:14.842 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m/tmp/tmpl9d3id6u[0m
    [35m2021-08-08 14:30:14.970 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m/tmp/tmpnnyxi_zg[0m
    [35m2021-08-08 14:30:15.743 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:30:15.743 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:30:15.744 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:30:16.140 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:30:16.141 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:30:17.046 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mWarning: you have pip-installed dependencies in your environment file, but you do not list pip itself as one of your conda dependencies.  Conda may not use the correct pip to install your packages, and they may end up in the wrong place.  Please add an explicit pip dependency.  I'm adding one for you, but still nagging you.[0m
    [35m2021-08-08 14:30:42.043 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting package metadata (repodata.json): ...working... done[0m
    [35m2021-08-08 14:30:43.235 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mSolving environment: ...working... done[0m
    [35m2021-08-08 14:30:43.340 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:30:43.709 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mPreparing transaction: ...working... done[0m
    [35m2021-08-08 14:30:44.380 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mVerifying transaction: ...working... done[0m
    [35m2021-08-08 14:30:45.091 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mExecuting transaction: ...working... done[0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mRan pip subprocess with arguments:[0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m['/home/clive/anaconda3/envs/tempo-38425be0-c06f-4783-bde2-95bf31bf1c5a/bin/python', '-m', 'pip', 'install', '-U', '-r', '/tmp/condaenv.axxcl2eo.requirements.txt'][0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mPip subprocess output:[0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting mlops-tempo[0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached mlops_tempo-0.3.0-py3-none-any.whl (74 kB)[0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting mlserver==0.3.2[0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached mlserver-0.3.2-py3-none-any.whl (46 kB)[0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting click[0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached click-8.0.1-py3-none-any.whl (97 kB)[0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting fastapi[0m
    [35m2021-08-08 14:30:58.904 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached fastapi-0.68.0-py3-none-any.whl (52 kB)[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting uvicorn[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached uvicorn-0.14.0-py3-none-any.whl (50 kB)[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting numpy[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached numpy-1.21.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.7 MB)[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting orjson[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached orjson-3.6.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (233 kB)[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting protobuf[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached protobuf-3.17.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting grpcio[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached grpcio-1.39.0-cp37-cp37m-manylinux2014_x86_64.whl (4.3 MB)[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting conda-pack[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached conda_pack-0.6.0-py2.py3-none-any.whl[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting packaging[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached packaging-21.0-py3-none-any.whl (40 kB)[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting aiohttp[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached aiohttp-3.7.4.post0-cp37-cp37m-manylinux2014_x86_64.whl (1.3 MB)[0m
    [35m2021-08-08 14:30:58.905 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting docker[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached docker-5.0.0-py2.py3-none-any.whl (146 kB)[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting requests[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached requests-2.26.0-py2.py3-none-any.whl (62 kB)[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting cloudpickle[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached cloudpickle-1.6.0-py3-none-any.whl (23 kB)[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting attrs[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached attrs-21.2.0-py2.py3-none-any.whl (53 kB)[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting python-rclone[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached python_rclone-0.0.2-py3-none-any.whl (4.2 kB)[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting redis[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached redis-3.5.3-py2.py3-none-any.whl (72 kB)[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting janus[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached janus-0.6.1-py3-none-any.whl (11 kB)[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting kubernetes[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached kubernetes-17.17.0-py3-none-any.whl (1.8 MB)[0m
    [35m2021-08-08 14:30:58.906 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting pydantic[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached pydantic-1.8.2-cp37-cp37m-manylinux2014_x86_64.whl (10.1 MB)[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting seldon-deploy-sdk[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached seldon_deploy_sdk-1.3.0-py3-none-any.whl (714 kB)[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting six>=1.5.2[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached six-1.16.0-py2.py3-none-any.whl (11 kB)[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting yarl<2.0,>=1.0[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached yarl-1.6.3-cp37-cp37m-manylinux2014_x86_64.whl (294 kB)[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting typing-extensions>=3.6.5[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached typing_extensions-3.10.0.0-py3-none-any.whl (26 kB)[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting chardet<5.0,>=2.0[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached chardet-4.0.0-py2.py3-none-any.whl (178 kB)[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting multidict<7.0,>=4.5[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached multidict-5.1.0-cp37-cp37m-manylinux2014_x86_64.whl (142 kB)[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting async-timeout<4.0,>=3.0[0m
    [35m2021-08-08 14:30:58.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached async_timeout-3.0.1-py3-none-any.whl (8.2 kB)[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting idna>=2.0[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached idna-3.2-py3-none-any.whl (59 kB)[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting importlib-metadata[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached importlib_metadata-4.6.3-py3-none-any.whl (17 kB)[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mRequirement already satisfied: setuptools in /home/clive/anaconda3/envs/tempo-38425be0-c06f-4783-bde2-95bf31bf1c5a/lib/python3.7/site-packages (from conda-pack->mlops-tempo->-r /tmp/condaenv.axxcl2eo.requirements.txt (line 1)) (52.0.0.post20210125)[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting websocket-client>=0.32.0[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached websocket_client-1.1.1-py2.py3-none-any.whl (68 kB)[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting urllib3<1.27,>=1.21.1[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached urllib3-1.26.6-py2.py3-none-any.whl (138 kB)[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting charset-normalizer~=2.0.0[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached charset_normalizer-2.0.4-py3-none-any.whl (36 kB)[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mRequirement already satisfied: certifi>=2017.4.17 in /home/clive/anaconda3/envs/tempo-38425be0-c06f-4783-bde2-95bf31bf1c5a/lib/python3.7/site-packages (from requests->mlops-tempo->-r /tmp/condaenv.axxcl2eo.requirements.txt (line 1)) (2021.5.30)[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting starlette==0.14.2[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached starlette-0.14.2-py3-none-any.whl (60 kB)[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting zipp>=0.5[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached zipp-3.5.0-py3-none-any.whl (5.7 kB)[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting google-auth>=1.0.1[0m
    [35m2021-08-08 14:30:58.908 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached google_auth-1.34.0-py2.py3-none-any.whl (152 kB)[0m
    [35m2021-08-08 14:30:58.909 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting pyyaml>=3.12[0m
    [35m2021-08-08 14:30:58.909 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)[0m
    [35m2021-08-08 14:30:58.909 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting requests-oauthlib[0m
    [35m2021-08-08 14:30:58.909 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)[0m
    [35m2021-08-08 14:30:58.909 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting python-dateutil>=2.5.3[0m
    [35m2021-08-08 14:31:00.874 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)[0m
    [35m2021-08-08 14:31:00.874 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting rsa<5,>=3.1.4[0m
    [35m2021-08-08 14:31:00.874 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached rsa-4.7.2-py3-none-any.whl (34 kB)[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting pyasn1-modules>=0.2.1[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting cachetools<5.0,>=2.0.0[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached cachetools-4.2.2-py3-none-any.whl (11 kB)[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting pyasn1<0.5.0,>=0.4.6[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting pyparsing>=2.0.2[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting oauthlib>=3.0.0[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached oauthlib-3.1.1-py2.py3-none-any.whl (146 kB)[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting Authlib<=0.16.0[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached Authlib-0.15.4-py2.py3-none-any.whl (203 kB)[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting cryptography[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached cryptography-3.4.7-cp36-abi3-manylinux2014_x86_64.whl (3.2 MB)[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting cffi>=1.12[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached cffi-1.14.6-cp37-cp37m-manylinux1_x86_64.whl (402 kB)[0m
    [35m2021-08-08 14:31:00.875 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting pycparser[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached pycparser-2.20-py2.py3-none-any.whl (112 kB)[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting h11>=0.8[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached h11-0.12.0-py3-none-any.whl (54 kB)[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting asgiref>=3.3.4[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mUsing cached asgiref-3.4.1-py3-none-any.whl (25 kB)[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mInstalling collected packages: zipp, typing-extensions, pycparser, urllib3, pyasn1, importlib-metadata, idna, charset-normalizer, cffi, starlette, six, rsa, requests, pydantic, pyasn1-modules, oauthlib, multidict, h11, cryptography, click, cachetools, asgiref, yarl, websocket-client, uvicorn, requests-oauthlib, pyyaml, python-dateutil, pyparsing, protobuf, orjson, numpy, grpcio, google-auth, fastapi, chardet, Authlib, attrs, async-timeout, seldon-deploy-sdk, redis, python-rclone, packaging, mlserver, kubernetes, janus, docker, conda-pack, cloudpickle, aiohttp, mlops-tempo[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mSuccessfully installed Authlib-0.15.4 aiohttp-3.7.4.post0 asgiref-3.4.1 async-timeout-3.0.1 attrs-21.2.0 cachetools-4.2.2 cffi-1.14.6 chardet-4.0.0 charset-normalizer-2.0.4 click-8.0.1 cloudpickle-1.6.0 conda-pack-0.6.0 cryptography-3.4.7 docker-5.0.0 fastapi-0.68.0 google-auth-1.34.0 grpcio-1.39.0 h11-0.12.0 idna-3.2 importlib-metadata-4.6.3 janus-0.6.1 kubernetes-17.17.0 mlops-tempo-0.3.0 mlserver-0.3.2 multidict-5.1.0 numpy-1.21.1 oauthlib-3.1.1 orjson-3.6.1 packaging-21.0 protobuf-3.17.3 pyasn1-0.4.8 pyasn1-modules-0.2.8 pycparser-2.20 pydantic-1.8.2 pyparsing-2.4.7 python-dateutil-2.8.2 python-rclone-0.0.2 pyyaml-5.4.1 redis-3.5.3 requests-2.26.0 requests-oauthlib-1.3.0 rsa-4.7.2 seldon-deploy-sdk-1.3.0 six-1.16.0 starlette-0.14.2 typing-extensions-3.10.0.0 urllib3-1.26.6 uvicorn-0.14.0 websocket-client-1.1.1 yarl-1.6.3 zipp-3.5.0[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m#[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m# To activate this environment, use[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m#[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m#     $ conda activate tempo-38425be0-c06f-4783-bde2-95bf31bf1c5a[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m#[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m# To deactivate an active environment, use[0m
    [35m2021-08-08 14:31:00.876 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m#[0m
    [35m2021-08-08 14:31:00.877 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m#     $ conda deactivate[0m
    [35m2021-08-08 14:31:00.877 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:00.877 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mCollecting packages...[0m
    [35m2021-08-08 14:31:01.316 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mPacking environment at '/home/clive/anaconda3/envs/tempo-38425be0-c06f-4783-bde2-95bf31bf1c5a' to '/tmp/tmp5_0ig7lj/environment.tar.gz'[0m
    [########################################] | 100% Completed |  9.7s[0m[22m[                                        ] | 0% Completed |  0.0s
    [35m2021-08-08 14:31:11.255 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.255 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m==> WARNING: A newer version of conda exists. <==[0m
    [35m2021-08-08 14:31:11.255 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mcurrent version: 4.7.12[0m
    [35m2021-08-08 14:31:11.255 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mlatest version: 4.10.3[0m
    [35m2021-08-08 14:31:11.255 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.255 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mPlease update conda by running[0m
    [35m2021-08-08 14:31:11.255 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.256 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m$ conda update -n base -c defaults conda[0m
    [35m2021-08-08 14:31:11.256 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.256 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.256 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.360 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m## Package Plan ##[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22menvironment location: /home/clive/anaconda3/envs/tempo-38425be0-c06f-4783-bde2-95bf31bf1c5a[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mThe following packages will be REMOVED:[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m_libgcc_mutex-0.1-main[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mca-certificates-2021.7.5-h06a4308_1[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mcertifi-2021.5.30-py37h06a4308_0[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mld_impl_linux-64-2.35.1-h7274673_9[0m
    [35m2021-08-08 14:31:11.361 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mlibffi-3.3-he6710b0_2[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mlibgcc-ng-9.1.0-hdf63c60_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mlibstdcxx-ng-9.1.0-hdf63c60_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mncurses-6.2-he6710b0_1[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mopenssl-1.1.1k-h27cfd23_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mpip-21.2.2-py37h06a4308_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mpython-3.7.9-h7579374_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mreadline-8.1-h27cfd23_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22msetuptools-52.0.0-py37h06a4308_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22msqlite-3.36.0-hc218d9a_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mtk-8.6.10-hbc83047_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mwheel-0.36.2-pyhd3eb1b0_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mxz-5.2.5-h7b6447c_0[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mzlib-1.2.11-h7b6447c_3[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.362 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:11.392 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mPreparing transaction: ...working... done[0m
    [35m2021-08-08 14:31:11.492 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mVerifying transaction: ...working... done[0m
    [35m2021-08-08 14:31:11.610 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mExecuting transaction: ...working... done[0m
    [35m2021-08-08 14:31:45.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mRemove all packages in environment /home/clive/anaconda3/envs/tempo-38425be0-c06f-4783-bde2-95bf31bf1c5a:[0m
    [35m2021-08-08 14:31:45.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m[0m
    [35m2021-08-08 14:31:45.907 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:31:55.955 [0m[32m[31/tempo/169 (pid 2468001)] [0m[22m{'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32), 'output1': 'xgboost prediction'}[0m
    [35m2021-08-08 15:32:00.792 [0m[32m[31/tempo/169 (pid 2468001)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:32:02.670 [0m[32m[31/end/170 (pid 2469464)] [0m[1mTask is starting.[0m
    [35m2021-08-08 15:32:06.082 [0m[32m[31/end/170 (pid 2469464)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:32:06.511 [0m[1mDone![0m


## Make Predictions with Metaflow Tempo Artifact


```python
from metaflow import Flow
run = Flow('IrisFlow').latest_run
```


```python
client = run.data.client_model
```


```python
import numpy as np
client.predict(np.array([[1, 2, 3, 4]]))
```




    {'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32),
     'output1': 'xgboost prediction'}



## Run Flow on AWS and Deploy to Remote Kubernetes

We will now run our flow on AWS Batch and will launch Tempo artifacts onto a remote Kubernetes cluster. make sure you have setup the cluster and have the appropriate authentication files as outlined above.

## Setup RBAC and Secret on Kubernetes Cluster


```python
!kubectl create ns production
```


```python
!kubectl create -f k8s/tempo-pipeline-rbac.yaml -n production
```

Create a Secret from the `k8s/s3_secret.yaml.tmpl` file by adding your AWS Key that can read from S3 and saving as `k8s/s3_secret.yaml`


```python
!kubectl create -f k8s/s3_secret.yaml -n production
```

## Run Metaflow on AWS Batch


```python
!python src/irisflow.py \
    --environment=conda \
    --with batch:image=seldonio/seldon-core-s2i-python37-ubi8:1.10.0-dev \
    run
```

    [35m[1mMetaflow 2.3.2[0m[35m[22m executing [0m[31m[1mIrisFlow[0m[35m[22m[0m[35m[22m for [0m[31m[1muser:clive[0m[35m[22m[K[0m[35m[22m[0m
    [35m[22mValidating your flow...[K[0m[35m[22m[0m
    [32m[1m    The graph looks good![K[0m[32m[1m[0m
    [35m[22mRunning pylint...[K[0m[35m[22m[0m
    [32m[22m    Pylint not found, so extra checks are disabled.[K[0m[32m[22m[0m
    [22mBootstrapping conda environment...(this could take a few minutes)[K[0m[22m[0m
    [22mIncluding file src/conda.yaml of size 115B [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-dq2lr7x0v0nz/metaflow/data/IrisFlow/a94f1cff7702ed70807d16917bb282f51a28511e[K[0m[22m[0m
    [22mIncluding file src/gsa-key.json of size 2KB [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-dq2lr7x0v0nz/metaflow/data/IrisFlow/6a24eaae697b14c2d8cdc2dcee1fce38b23b6ce0[K[0m[22m[0m
    [22mIncluding file src/kubeconfig.yaml of size 2KB [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-dq2lr7x0v0nz/metaflow/data/IrisFlow/00e21f5423ea21f695831e970aa81a283ac25812[K[0m[22m[0m
    [35m2021-08-08 15:33:21.881 [0m[1mWorkflow starting (run-id 32):[0m
    [35m2021-08-08 15:33:23.615 [0m[32m[32/start/172 (pid 2469999)] [0m[1mTask is starting.[0m
    [35m2021-08-08 14:33:24.695 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-08 14:33:30.189 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-08 14:33:35.635 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Task is starting (status STARTING)...[0m
    [35m2021-08-08 14:34:02.032 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Task is starting (status RUNNING)...[0m
    [35m2021-08-08 14:34:01.085 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Setting up task environment.[0m
    [35m2021-08-08 14:34:07.172 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Downloading code package...[0m
    [35m2021-08-08 14:34:07.679 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Code package downloaded.[0m
    [35m2021-08-08 14:34:07.691 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Task is starting.[0m
    [35m2021-08-08 14:34:08.057 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Bootstrapping environment...[0m
    [35m2021-08-08 14:34:32.229 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Environment bootstrapped.[0m
    [35m2021-08-08 14:34:42.634 [0m[32m[32/start/172 (pid 2469999)] [0m[22m[aa141922-797f-4e1e-8216-49b77312a969] Task finished with exit code 0.[0m
    [35m2021-08-08 15:34:43.571 [0m[32m[32/start/172 (pid 2469999)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:34:45.443 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[1mTask is starting.[0m
    [35m2021-08-08 15:34:46.764 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[1mTask is starting.[0m
    [35m2021-08-08 14:34:46.764 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-08 14:34:47.885 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-08 14:34:48.706 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-08 14:34:48.983 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-08 14:34:49.792 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Task is starting (status STARTING)...[0m
    [35m2021-08-08 14:34:50.067 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Task is starting (status STARTING)...[0m
    [35m2021-08-08 14:34:53.343 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Task is starting (status RUNNING)...[0m
    [35m2021-08-08 14:34:54.142 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Task is starting (status RUNNING)...[0m
    [35m2021-08-08 14:34:52.064 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Setting up task environment.[0m
    [35m2021-08-08 14:34:58.234 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Downloading code package...[0m
    [35m2021-08-08 14:34:52.077 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Setting up task environment.[0m
    [35m2021-08-08 14:34:58.207 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Downloading code package...[0m
    [35m2021-08-08 14:34:58.726 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Code package downloaded.[0m
    [35m2021-08-08 14:34:58.738 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Task is starting.[0m
    [35m2021-08-08 14:34:59.120 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Bootstrapping environment...[0m
    [35m2021-08-08 14:35:26.002 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Environment bootstrapped.[0m
    [35m2021-08-08 14:35:27.470 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] [14:35:27] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.[0m
    [35m2021-08-08 14:35:36.228 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[22m[13dfaaa8-27bb-4f00-a2e3-7deb4f8ad038] Task finished with exit code 0.[0m
    [35m2021-08-08 15:35:37.077 [0m[32m[32/train_xgboost/174 (pid 2470544)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 14:34:58.762 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Code package downloaded.[0m
    [35m2021-08-08 14:34:58.777 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Task is starting.[0m
    [35m2021-08-08 14:34:59.242 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Bootstrapping environment...[0m
    [35m2021-08-08 14:35:25.772 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Environment bootstrapped.[0m
    [35m2021-08-08 14:35:37.628 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[22m[0c0c0dcf-6e75-4518-ac39-69b8acdb9c4f] Task finished with exit code 0.[0m
    [35m2021-08-08 15:35:38.143 [0m[32m[32/train_sklearn/173 (pid 2470525)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:35:39.209 [0m[32m[32/join/175 (pid 2470870)] [0m[1mTask is starting.[0m
    [35m2021-08-08 14:35:40.355 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-08 14:35:42.535 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-08 14:35:43.627 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Task is starting (status STARTING)...[0m
    [35m2021-08-08 14:35:46.905 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Task is starting (status RUNNING)...[0m
    [35m2021-08-08 14:35:46.039 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Setting up task environment.[0m
    [35m2021-08-08 14:35:52.108 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Downloading code package...[0m
    [35m2021-08-08 14:35:52.612 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Code package downloaded.[0m
    [35m2021-08-08 14:35:52.622 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Task is starting.[0m
    [35m2021-08-08 14:35:52.987 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Bootstrapping environment...[0m
    [35m2021-08-08 14:36:09.321 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Environment bootstrapped.[0m
    [35m2021-08-08 14:36:18.563 [0m[32m[32/join/175 (pid 2470870)] [0m[22m[36331d73-c958-4aa7-92a9-c0f3b6258d47] Task finished with exit code 0.[0m
    [35m2021-08-08 15:36:19.404 [0m[32m[32/join/175 (pid 2470870)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:36:21.203 [0m[32m[32/tempo/176 (pid 2471152)] [0m[1mTask is starting.[0m
    [35m2021-08-08 14:36:22.364 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-08 14:36:25.652 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-08 14:36:31.100 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Task is starting (status STARTING)...[0m
    [35m2021-08-08 14:36:34.413 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Task is starting (status RUNNING)...[0m
    [35m2021-08-08 14:36:33.061 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Setting up task environment.[0m
    [35m2021-08-08 14:36:39.204 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Downloading code package...[0m
    [35m2021-08-08 14:36:39.709 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Code package downloaded.[0m
    [35m2021-08-08 14:36:39.721 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Task is starting.[0m
    [35m2021-08-08 14:36:40.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Bootstrapping environment...[0m
    [35m2021-08-08 14:36:59.191 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Environment bootstrapped.[0m
    [35m2021-08-08 14:36:59.845 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Pip Test Install: mlops-tempo 0.4.0.dev5[0m
    [35m2021-08-08 14:37:19.297 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Pip Test Install: conda_env 2.4.2[0m
    [35m2021-08-08 14:37:18.814 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m
    [35m2021-08-08 14:37:21.299 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] /tmp/tmpr4v9ouza[0m
    [35m2021-08-08 14:37:21.337 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] /tmp/tmppelyoow0[0m
    [35m2021-08-08 14:37:20.626 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m
    [35m2021-08-08 14:37:22.166 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Warning: you have pip-installed dependencies in your environment file, but you do not list pip itself as one of your conda dependencies.  Conda may not use the correct pip to install your packages, and they may end up in the wrong place.  Please add an explicit pip dependency.  I'm adding one for you, but still nagging you.[0m
    [35m2021-08-08 14:37:21.564 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:37:21.565 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:37:21.565 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:37:21.692 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:37:21.692 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:37:26.003 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting package metadata (repodata.json): ...working... done[0m
    [35m2021-08-08 14:37:26.113 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Solving environment: ...working... done[0m
    [35m2021-08-08 14:37:26.243 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:37:26.243 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Downloading and Extracting Packages[0m
    wheel-0.36.2         | 33 KB     | ########## | 100%[0m 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] 
    pip-21.2.2           | 1.8 MB    | ########## | 100%[0m 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] 
    _libgcc_mutex-0.1    | 3 KB      | ########## | 100%[0m 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] 
    setuptools-52.0.0    | 710 KB    | ########## | 100%[0m 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] 
    [35m2021-08-08 14:37:26.213 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:37:26.213 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:37:26.213 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] ==> WARNING: A newer version of conda exists. <==[0m
    [35m2021-08-08 14:37:26.213 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   current version: 4.7.12[0m
    [35m2021-08-08 14:37:26.213 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   latest version: 4.10.3[0m
    [35m2021-08-08 14:37:26.214 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:37:26.214 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Please update conda by running[0m
    [35m2021-08-08 14:37:26.214 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:37:26.214 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]     $ conda update -n base -c defaults conda[0m
    [35m2021-08-08 14:37:26.214 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:37:26.214 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    python-3.7.9         | 45.3 MB   | ########## | 100%[0m 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] 
    [35m2021-08-08 14:37:27.713 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Preparing transaction: ...working... done[0m
    [35m2021-08-08 14:37:28.566 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Verifying transaction: ...working... done[0m
    [35m2021-08-08 14:37:36.195 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Executing transaction: ...working... done[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Ran pip subprocess with arguments:[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] ['/opt/conda/envs/tempo-3b5a29db-9160-4897-89c8-3e5a250f8186/bin/python', '-m', 'pip', 'install', '-U', '-r', '/tmp/condaenv.idvat6xa.requirements.txt'][0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Pip subprocess output:[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting mlops-tempo[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading mlops_tempo-0.3.0-py3-none-any.whl (74 kB)[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting mlserver==0.3.2[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading mlserver-0.3.2-py3-none-any.whl (46 kB)[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting numpy[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached numpy-1.21.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.7 MB)[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting fastapi[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached fastapi-0.68.0-py3-none-any.whl (52 kB)[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting protobuf[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached protobuf-3.17.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting grpcio[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached grpcio-1.39.0-cp37-cp37m-manylinux2014_x86_64.whl (4.3 MB)[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting uvicorn[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached uvicorn-0.14.0-py3-none-any.whl (50 kB)[0m
    [35m2021-08-08 14:37:51.091 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting click[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached click-8.0.1-py3-none-any.whl (97 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting orjson[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached orjson-3.6.1-cp37-cp37m-manylinux_2_24_x86_64.whl (233 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting attrs[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached attrs-21.2.0-py2.py3-none-any.whl (53 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting pydantic[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached pydantic-1.8.2-cp37-cp37m-manylinux2014_x86_64.whl (10.1 MB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting packaging[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached packaging-21.0-py3-none-any.whl (40 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting seldon-deploy-sdk[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached seldon_deploy_sdk-1.3.0-py3-none-any.whl (714 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting redis[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached redis-3.5.3-py2.py3-none-any.whl (72 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting cloudpickle[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached cloudpickle-1.6.0-py3-none-any.whl (23 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting python-rclone[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached python_rclone-0.0.2-py3-none-any.whl (4.2 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting janus[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached janus-0.6.1-py3-none-any.whl (11 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting conda-pack[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached conda_pack-0.6.0-py2.py3-none-any.whl[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting kubernetes[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached kubernetes-17.17.0-py3-none-any.whl (1.8 MB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting requests[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading requests-2.26.0-py2.py3-none-any.whl (62 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting aiohttp[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached aiohttp-3.7.4.post0-cp37-cp37m-manylinux2014_x86_64.whl (1.3 MB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting docker[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached docker-5.0.0-py2.py3-none-any.whl (146 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting six>=1.5.2[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)[0m
    [35m2021-08-08 14:37:51.092 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting multidict<7.0,>=4.5[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached multidict-5.1.0-cp37-cp37m-manylinux2014_x86_64.whl (142 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting yarl<2.0,>=1.0[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached yarl-1.6.3-cp37-cp37m-manylinux2014_x86_64.whl (294 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting chardet<5.0,>=2.0[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading chardet-4.0.0-py2.py3-none-any.whl (178 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting typing-extensions>=3.6.5[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached typing_extensions-3.10.0.0-py3-none-any.whl (26 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting async-timeout<4.0,>=3.0[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached async_timeout-3.0.1-py3-none-any.whl (8.2 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting idna>=2.0[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading idna-3.2-py3-none-any.whl (59 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting importlib-metadata[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached importlib_metadata-4.6.3-py3-none-any.whl (17 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Requirement already satisfied: setuptools in /opt/conda/envs/tempo-3b5a29db-9160-4897-89c8-3e5a250f8186/lib/python3.7/site-packages (from conda-pack->mlops-tempo->-r /tmp/condaenv.idvat6xa.requirements.txt (line 1)) (52.0.0.post20210125)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting websocket-client>=0.32.0[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached websocket_client-1.1.1-py2.py3-none-any.whl (68 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/tempo-3b5a29db-9160-4897-89c8-3e5a250f8186/lib/python3.7/site-packages (from requests->mlops-tempo->-r /tmp/condaenv.idvat6xa.requirements.txt (line 1)) (2021.5.30)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting charset-normalizer~=2.0.0[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading charset_normalizer-2.0.4-py3-none-any.whl (36 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting urllib3<1.27,>=1.21.1[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading urllib3-1.26.6-py2.py3-none-any.whl (138 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting starlette==0.14.2[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached starlette-0.14.2-py3-none-any.whl (60 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting zipp>=0.5[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached zipp-3.5.0-py3-none-any.whl (5.7 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting requests-oauthlib[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)[0m
    [35m2021-08-08 14:37:51.093 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting google-auth>=1.0.1[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached google_auth-1.34.0-py2.py3-none-any.whl (152 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting python-dateutil>=2.5.3[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting pyyaml>=3.12[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting cachetools<5.0,>=2.0.0[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached cachetools-4.2.2-py3-none-any.whl (11 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting pyasn1-modules>=0.2.1[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting rsa<5,>=3.1.4[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached rsa-4.7.2-py3-none-any.whl (34 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting pyasn1<0.5.0,>=0.4.6[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting pyparsing>=2.0.2[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting oauthlib>=3.0.0[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached oauthlib-3.1.1-py2.py3-none-any.whl (146 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting Authlib<=0.16.0[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached Authlib-0.15.4-py2.py3-none-any.whl (203 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting cryptography[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading cryptography-3.4.7-cp36-abi3-manylinux2014_x86_64.whl (3.2 MB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting cffi>=1.12[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading cffi-1.14.6-cp37-cp37m-manylinux1_x86_64.whl (402 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting pycparser[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Downloading pycparser-2.20-py2.py3-none-any.whl (112 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting h11>=0.8[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached h11-0.12.0-py3-none-any.whl (54 kB)[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting asgiref>=3.3.4[0m
    [35m2021-08-08 14:37:51.094 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   Using cached asgiref-3.4.1-py3-none-any.whl (25 kB)[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Installing collected packages: zipp, typing-extensions, pycparser, urllib3, pyasn1, importlib-metadata, idna, charset-normalizer, cffi, starlette, six, rsa, requests, pydantic, pyasn1-modules, oauthlib, multidict, h11, cryptography, click, cachetools, asgiref, yarl, websocket-client, uvicorn, requests-oauthlib, pyyaml, python-dateutil, pyparsing, protobuf, orjson, numpy, grpcio, google-auth, fastapi, chardet, Authlib, attrs, async-timeout, seldon-deploy-sdk, redis, python-rclone, packaging, mlserver, kubernetes, janus, docker, conda-pack, cloudpickle, aiohttp, mlops-tempo[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Successfully installed Authlib-0.15.4 aiohttp-3.7.4.post0 asgiref-3.4.1 async-timeout-3.0.1 attrs-21.2.0 cachetools-4.2.2 cffi-1.14.6 chardet-4.0.0 charset-normalizer-2.0.4 click-8.0.1 cloudpickle-1.6.0 conda-pack-0.6.0 cryptography-3.4.7 docker-5.0.0 fastapi-0.68.0 google-auth-1.34.0 grpcio-1.39.0 h11-0.12.0 idna-3.2 importlib-metadata-4.6.3 janus-0.6.1 kubernetes-17.17.0 mlops-tempo-0.3.0 mlserver-0.3.2 multidict-5.1.0 numpy-1.21.1 oauthlib-3.1.1 orjson-3.6.1 packaging-21.0 protobuf-3.17.3 pyasn1-0.4.8 pyasn1-modules-0.2.8 pycparser-2.20 pydantic-1.8.2 pyparsing-2.4.7 python-dateutil-2.8.2 python-rclone-0.0.2 pyyaml-5.4.1 redis-3.5.3 requests-2.26.0 requests-oauthlib-1.3.0 rsa-4.7.2 seldon-deploy-sdk-1.3.0 six-1.16.0 starlette-0.14.2 typing-extensions-3.10.0.0 urllib3-1.26.6 uvicorn-0.14.0 websocket-client-1.1.1 yarl-1.6.3 zipp-3.5.0[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] #[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] # To activate this environment, use[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] #[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] #     $ conda activate tempo-3b5a29db-9160-4897-89c8-3e5a250f8186[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] #[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] # To deactivate an active environment, use[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] #[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] #     $ conda deactivate[0m
    [35m2021-08-08 14:37:51.095 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:37:51.444 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Collecting packages...[0m
    [35m2021-08-08 14:37:52.022 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Packing environment at '/opt/conda/envs/tempo-3b5a29db-9160-4897-89c8-3e5a250f8186' to '/tmp/tmpu0omqa66/environment.tar.gz'[0m
    [########################################] | 100% Completed | 10.9s[0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] 
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] ## Package Plan ##[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   environment location: /opt/conda/envs/tempo-3b5a29db-9160-4897-89c8-3e5a250f8186[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] The following packages will be REMOVED:[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   _libgcc_mutex-0.1-main[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   ca-certificates-2021.7.5-h06a4308_1[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   certifi-2021.5.30-py37h06a4308_0[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   ld_impl_linux-64-2.35.1-h7274673_9[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   libffi-3.3-he6710b0_2[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   libgcc-ng-9.1.0-hdf63c60_0[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   libstdcxx-ng-9.1.0-hdf63c60_0[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   ncurses-6.2-he6710b0_1[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   openssl-1.1.1k-h27cfd23_0[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   pip-21.2.2-py37h06a4308_0[0m
    [35m2021-08-08 14:38:03.372 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   python-3.7.9-h7579374_0[0m
    [35m2021-08-08 14:38:03.373 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   readline-8.1-h27cfd23_0[0m
    [35m2021-08-08 14:38:03.373 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   setuptools-52.0.0-py37h06a4308_0[0m
    [35m2021-08-08 14:38:03.373 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   sqlite-3.36.0-hc218d9a_0[0m
    [35m2021-08-08 14:38:03.373 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   tk-8.6.10-hbc83047_0[0m
    [35m2021-08-08 14:38:03.373 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   wheel-0.36.2-pyhd3eb1b0_0[0m
    [35m2021-08-08 14:38:03.373 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   xz-5.2.5-h7b6447c_0[0m
    [35m2021-08-08 14:38:03.373 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb]   zlib-1.2.11-h7b6447c_3[0m
    [35m2021-08-08 14:38:03.373 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:38:03.373 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:38:03.407 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Preparing transaction: ...working... done[0m
    [35m2021-08-08 14:38:03.504 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Verifying transaction: ...working... done[0m
    [35m2021-08-08 14:38:03.699 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Executing transaction: ...working... done[0m
    [35m2021-08-08 14:38:03.269 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:38:03.270 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Remove all packages in environment /opt/conda/envs/tempo-3b5a29db-9160-4897-89c8-3e5a250f8186:[0m
    [35m2021-08-08 14:38:03.270 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb][0m
    [35m2021-08-08 14:38:08.608 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-08 14:38:18.854 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] {'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32), 'output1': 'xgboost prediction'}[0m
    [35m2021-08-08 14:38:23.019 [0m[32m[32/tempo/176 (pid 2471152)] [0m[22m[71cbd837-9cf1-43a8-bdb9-e2c898f4f3eb] Task finished with exit code 0.[0m
    [35m2021-08-08 15:38:23.844 [0m[32m[32/tempo/176 (pid 2471152)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:38:24.892 [0m[32m[32/end/177 (pid 2471980)] [0m[1mTask is starting.[0m
    [35m2021-08-08 14:38:25.925 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-08 14:38:28.110 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-08 14:38:31.374 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Task is starting (status STARTING)...[0m
    [35m2021-08-08 14:38:34.664 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Task is starting (status RUNNING)...[0m
    [35m2021-08-08 14:38:32.954 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Setting up task environment.[0m
    [35m2021-08-08 14:38:39.092 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Downloading code package...[0m
    [35m2021-08-08 14:38:39.591 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Code package downloaded.[0m
    [35m2021-08-08 14:38:39.602 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Task is starting.[0m
    [35m2021-08-08 14:38:39.964 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Bootstrapping environment...[0m
    [35m2021-08-08 14:38:56.230 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Environment bootstrapped.[0m
    [35m2021-08-08 14:39:04.503 [0m[32m[32/end/177 (pid 2471980)] [0m[22m[d6392d28-0aec-416f-ac0e-670e0ef63c0b] Task finished with exit code 0.[0m
    [35m2021-08-08 15:39:05.339 [0m[32m[32/end/177 (pid 2471980)] [0m[1mTask finished successfully.[0m
    [35m2021-08-08 15:39:05.789 [0m[1mDone![0m


## Make Predictions with Metaflow Tempo Artifact


```python
from metaflow import Flow
run = Flow('IrisFlow').latest_run
```


```python
client = run.data.client_model
```


```python
import numpy as np
client.predict(np.array([[1, 2, 3, 4]]))
```




    {'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32),
     'output1': 'xgboost prediction'}



## Tempo Utils for creating your own Metaflow workflow


```python
import tempo.metaflow.utils
```


```python
%pdoc tempo.metaflow.utils.save_artifact
```


```python

```
