# End to End ML with Metaflow and Tempo

We will train two models and deploy them with tempo within a Metaflow pipeline. To understand the core example see [here](https://tempo.readthedocs.io/en/latest/examples/multi-model/README.html)

![archtecture](architecture.png)

## MetaFlow Prequisites


### Install metaflow locally

```
pip install metaflow
```

### Setup Conda-Forge Support

The flow will use conda-forge so you need to add that channel to conda.

```
conda config --add channels conda-forge
```



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


## Run Flow locally to deploy to Docker

To run the workflow with a local Docker deployment use the flag:

```
--tempo-on-docker true
```



```python
!python src/irisflow.py --environment=conda run 
```

    [35m[1mMetaflow 2.3.2[0m[35m[22m executing [0m[31m[1mIrisFlow[0m[35m[22m[0m[35m[22m for [0m[31m[1muser:clive[0m[35m[22m[K[0m[35m[22m[0m
    [35m[22mValidating your flow...[K[0m[35m[22m[0m
    [32m[1m    The graph looks good![K[0m[32m[1m[0m
    [35m[22mRunning pylint...[K[0m[35m[22m[0m
    [32m[22m    Pylint not found, so extra checks are disabled.[K[0m[32m[22m[0m
    [22mBootstrapping conda environment...(this could take a few minutes)[K[0m[22m[0m
    [22mIncluding file src/conda.yaml of size 115B [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-1ou61547mcwbq/metaflow/data/IrisFlow/a94f1cff7702ed70807d16917bb282f51a28511e[K[0m[22m[0m
    [22mIncluding file src/gsa-key.json of size 2KB [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-1ou61547mcwbq/metaflow/data/IrisFlow/c39c6e4ed18f183a2a3ae16b491f2848b57f6824[K[0m[22m[0m
    [22mIncluding file src/kubeconfig.yaml of size 2KB [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-1ou61547mcwbq/metaflow/data/IrisFlow/a203db53bd80aea425e9fdc206db3d107ce77f69[K[0m[22m[0m
    [35m2021-09-04 09:15:26.734 [0m[1mWorkflow starting (run-id 2):[0m
    [35m2021-09-04 09:15:28.219 [0m[32m[2/start/9 (pid 535320)] [0m[1mTask is starting.[0m
    [35m2021-09-04 09:15:32.460 [0m[32m[2/start/9 (pid 535320)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 09:15:34.087 [0m[32m[2/train_sklearn/10 (pid 535414)] [0m[1mTask is starting.[0m
    [35m2021-09-04 09:15:35.157 [0m[32m[2/train_xgboost/11 (pid 535438)] [0m[1mTask is starting.[0m
    [35m2021-09-04 08:15:37.523 [0m[32m[2/train_xgboost/11 (pid 535438)] [0m[22m[09:15:37] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.[0m
    [35m2021-09-04 09:15:38.486 [0m[32m[2/train_sklearn/10 (pid 535414)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 09:15:39.565 [0m[32m[2/train_xgboost/11 (pid 535438)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 09:15:41.217 [0m[32m[2/join/12 (pid 535587)] [0m[1mTask is starting.[0m
    [35m2021-09-04 09:15:46.874 [0m[32m[2/join/12 (pid 535587)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 09:15:48.454 [0m[32m[2/tempo/13 (pid 535666)] [0m[1mTask is starting.[0m
    [35m2021-09-04 08:15:50.337 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mPip Test Install: mlops-tempo 0.5.0[0m
    [35m2021-09-04 08:16:18.237 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mPip Test Install: conda_env 2.4.2[0m
    [35m2021-09-04 08:16:20.460 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m/tmp/tmpueb2915s[0m
    [35m2021-09-04 08:16:20.583 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m/tmp/tmpzw7ojyxo[0m
    [35m2021-09-04 08:16:21.304 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 08:16:21.304 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 08:16:21.305 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 08:16:21.719 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 08:16:21.719 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 08:16:23.314 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mWarning: you have pip-installed dependencies in your environment file, but you do not list pip itself as one of your conda dependencies.  Conda may not use the correct pip to install your packages, and they may end up in the wrong place.  Please add an explicit pip dependency.  I'm adding one for you, but still nagging you.[0m
    [35m2021-09-04 08:16:59.665 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting package metadata (repodata.json): ...working... done[0m
    [35m2021-09-04 08:17:01.573 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mSolving environment: ...working... done[0m
    [35m2021-09-04 08:17:01.823 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:02.367 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:02.367 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mDownloading and Extracting Packages[0m
    openssl-1.1.1l       | 3.8 MB    | ########## | 100%[0m35666)] [0m[22mopenssl-1.1.1l       | 3.8 MB    |            |   0% 
    [35m2021-09-04 08:17:03.677 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mPreparing transaction: ...working... done[0m
    [35m2021-09-04 08:17:04.938 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mVerifying transaction: ...working... done[0m
    [35m2021-09-04 08:17:06.371 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mExecuting transaction: ...working... done[0m
    [35m2021-09-04 08:17:26.184 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mRan pip subprocess with arguments:[0m
    [35m2021-09-04 08:17:26.185 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m['/home/clive/anaconda3/envs/tempo-bfe7ae29-ca47-4b67-a749-8c99f97d780e/bin/python', '-m', 'pip', 'install', '-U', '-r', '/tmp/condaenv.rnh52e10.requirements.txt'][0m
    [35m2021-09-04 08:17:26.185 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mPip subprocess output:[0m
    [35m2021-09-04 08:17:26.185 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting mlops-tempo[0m
    [35m2021-09-04 08:17:26.185 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached mlops_tempo-0.3.0-py3-none-any.whl (74 kB)[0m
    [35m2021-09-04 08:17:26.185 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting mlserver==0.3.2[0m
    [35m2021-09-04 08:17:26.186 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached mlserver-0.3.2-py3-none-any.whl (46 kB)[0m
    [35m2021-09-04 08:17:26.186 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting fastapi[0m
    [35m2021-09-04 08:17:26.186 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached fastapi-0.68.1-py3-none-any.whl (52 kB)[0m
    [35m2021-09-04 08:17:26.186 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting numpy[0m
    [35m2021-09-04 08:17:26.186 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached numpy-1.21.2-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.7 MB)[0m
    [35m2021-09-04 08:17:26.186 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting orjson[0m
    [35m2021-09-04 08:17:26.186 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached orjson-3.6.3-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (234 kB)[0m
    [35m2021-09-04 08:17:26.186 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting protobuf[0m
    [35m2021-09-04 08:17:26.186 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached protobuf-3.17.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)[0m
    [35m2021-09-04 08:17:26.186 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting grpcio[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached grpcio-1.39.0-cp37-cp37m-manylinux2014_x86_64.whl (4.3 MB)[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting uvicorn[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached uvicorn-0.15.0-py3-none-any.whl (54 kB)[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting click[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached click-8.0.1-py3-none-any.whl (97 kB)[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting redis[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached redis-3.5.3-py2.py3-none-any.whl (72 kB)[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting kubernetes[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached kubernetes-18.20.0-py2.py3-none-any.whl (1.6 MB)[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting python-rclone[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached python_rclone-0.0.2-py3-none-any.whl (4.2 kB)[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting seldon-deploy-sdk[0m
    [35m2021-09-04 08:17:26.187 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached seldon_deploy_sdk-1.3.0-py3-none-any.whl (714 kB)[0m
    [35m2021-09-04 08:17:26.188 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting cloudpickle[0m
    [35m2021-09-04 08:17:26.188 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached cloudpickle-1.6.0-py3-none-any.whl (23 kB)[0m
    [35m2021-09-04 08:17:26.188 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting docker[0m
    [35m2021-09-04 08:17:26.188 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached docker-5.0.2-py2.py3-none-any.whl (145 kB)[0m
    [35m2021-09-04 08:17:26.188 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting conda-pack[0m
    [35m2021-09-04 08:17:26.188 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached conda_pack-0.6.0-py2.py3-none-any.whl[0m
    [35m2021-09-04 08:17:26.188 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting packaging[0m
    [35m2021-09-04 08:17:26.188 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached packaging-21.0-py3-none-any.whl (40 kB)[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting janus[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached janus-0.6.1-py3-none-any.whl (11 kB)[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting attrs[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached attrs-21.2.0-py2.py3-none-any.whl (53 kB)[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting pydantic[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached pydantic-1.8.2-cp37-cp37m-manylinux2014_x86_64.whl (10.1 MB)[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting requests[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached requests-2.26.0-py2.py3-none-any.whl (62 kB)[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting aiohttp[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached aiohttp-3.7.4.post0-cp37-cp37m-manylinux2014_x86_64.whl (1.3 MB)[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting six>=1.5.2[0m
    [35m2021-09-04 08:17:26.189 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached six-1.16.0-py2.py3-none-any.whl (11 kB)[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting typing-extensions>=3.6.5[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached typing_extensions-3.10.0.2-py3-none-any.whl (26 kB)[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting async-timeout<4.0,>=3.0[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached async_timeout-3.0.1-py3-none-any.whl (8.2 kB)[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting yarl<2.0,>=1.0[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached yarl-1.6.3-cp37-cp37m-manylinux2014_x86_64.whl (294 kB)[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting multidict<7.0,>=4.5[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached multidict-5.1.0-cp37-cp37m-manylinux2014_x86_64.whl (142 kB)[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting chardet<5.0,>=2.0[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached chardet-4.0.0-py2.py3-none-any.whl (178 kB)[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting idna>=2.0[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached idna-3.2-py3-none-any.whl (59 kB)[0m
    [35m2021-09-04 08:17:26.190 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting importlib-metadata[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mDownloading importlib_metadata-4.8.1-py3-none-any.whl (17 kB)[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mRequirement already satisfied: setuptools in /home/clive/anaconda3/envs/tempo-bfe7ae29-ca47-4b67-a749-8c99f97d780e/lib/python3.7/site-packages (from conda-pack->mlops-tempo->-r /tmp/condaenv.rnh52e10.requirements.txt (line 1)) (52.0.0.post20210125)[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting websocket-client>=0.32.0[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached websocket_client-1.2.1-py2.py3-none-any.whl (52 kB)[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mRequirement already satisfied: certifi>=2017.4.17 in /home/clive/anaconda3/envs/tempo-bfe7ae29-ca47-4b67-a749-8c99f97d780e/lib/python3.7/site-packages (from requests->mlops-tempo->-r /tmp/condaenv.rnh52e10.requirements.txt (line 1)) (2021.5.30)[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting urllib3<1.27,>=1.21.1[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached urllib3-1.26.6-py2.py3-none-any.whl (138 kB)[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting charset-normalizer~=2.0.0[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached charset_normalizer-2.0.4-py3-none-any.whl (36 kB)[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting starlette==0.14.2[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached starlette-0.14.2-py3-none-any.whl (60 kB)[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting zipp>=0.5[0m
    [35m2021-09-04 08:17:26.191 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached zipp-3.5.0-py3-none-any.whl (5.7 kB)[0m
    [35m2021-09-04 08:17:26.192 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting python-dateutil>=2.5.3[0m
    [35m2021-09-04 08:17:26.192 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)[0m
    [35m2021-09-04 08:17:26.192 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting pyyaml>=5.4.1[0m
    [35m2021-09-04 08:17:26.192 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)[0m
    [35m2021-09-04 08:17:26.192 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting google-auth>=1.0.1[0m
    [35m2021-09-04 08:17:26.192 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached google_auth-2.0.2-py2.py3-none-any.whl (152 kB)[0m
    [35m2021-09-04 08:17:26.192 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting requests-oauthlib[0m
    [35m2021-09-04 08:17:28.850 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)[0m
    [35m2021-09-04 08:17:28.850 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting cachetools<5.0,>=2.0.0[0m
    [35m2021-09-04 08:17:28.851 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached cachetools-4.2.2-py3-none-any.whl (11 kB)[0m
    [35m2021-09-04 08:17:28.851 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting pyasn1-modules>=0.2.1[0m
    [35m2021-09-04 08:17:28.851 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)[0m
    [35m2021-09-04 08:17:28.853 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting rsa<5,>=3.1.4[0m
    [35m2021-09-04 08:17:28.853 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached rsa-4.7.2-py3-none-any.whl (34 kB)[0m
    [35m2021-09-04 08:17:28.853 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting pyasn1<0.5.0,>=0.4.6[0m
    [35m2021-09-04 08:17:28.853 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)[0m
    [35m2021-09-04 08:17:28.853 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting pyparsing>=2.0.2[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting oauthlib>=3.0.0[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached oauthlib-3.1.1-py2.py3-none-any.whl (146 kB)[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting Authlib<=0.16.0[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached Authlib-0.15.4-py2.py3-none-any.whl (203 kB)[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting cryptography[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached cryptography-3.4.8-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2 MB)[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting cffi>=1.12[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached cffi-1.14.6-cp37-cp37m-manylinux1_x86_64.whl (402 kB)[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting pycparser[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached pycparser-2.20-py2.py3-none-any.whl (112 kB)[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting h11>=0.8[0m
    [35m2021-09-04 08:17:28.854 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached h11-0.12.0-py3-none-any.whl (54 kB)[0m
    [35m2021-09-04 08:17:28.855 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting asgiref>=3.4.0[0m
    [35m2021-09-04 08:17:28.855 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mUsing cached asgiref-3.4.1-py3-none-any.whl (25 kB)[0m
    [35m2021-09-04 08:17:28.855 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mInstalling collected packages: zipp, typing-extensions, pycparser, urllib3, pyasn1, importlib-metadata, idna, charset-normalizer, cffi, starlette, six, rsa, requests, pydantic, pyasn1-modules, oauthlib, multidict, h11, cryptography, click, cachetools, asgiref, yarl, websocket-client, uvicorn, requests-oauthlib, pyyaml, python-dateutil, pyparsing, protobuf, orjson, numpy, grpcio, google-auth, fastapi, chardet, Authlib, attrs, async-timeout, seldon-deploy-sdk, redis, python-rclone, packaging, mlserver, kubernetes, janus, docker, conda-pack, cloudpickle, aiohttp, mlops-tempo[0m
    [35m2021-09-04 08:17:28.855 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mSuccessfully installed Authlib-0.15.4 aiohttp-3.7.4.post0 asgiref-3.4.1 async-timeout-3.0.1 attrs-21.2.0 cachetools-4.2.2 cffi-1.14.6 chardet-4.0.0 charset-normalizer-2.0.4 click-8.0.1 cloudpickle-1.6.0 conda-pack-0.6.0 cryptography-3.4.8 docker-5.0.2 fastapi-0.68.1 google-auth-2.0.2 grpcio-1.39.0 h11-0.12.0 idna-3.2 importlib-metadata-4.8.1 janus-0.6.1 kubernetes-18.20.0 mlops-tempo-0.3.0 mlserver-0.3.2 multidict-5.1.0 numpy-1.21.2 oauthlib-3.1.1 orjson-3.6.3 packaging-21.0 protobuf-3.17.3 pyasn1-0.4.8 pyasn1-modules-0.2.8 pycparser-2.20 pydantic-1.8.2 pyparsing-2.4.7 python-dateutil-2.8.2 python-rclone-0.0.2 pyyaml-5.4.1 redis-3.5.3 requests-2.26.0 requests-oauthlib-1.3.0 rsa-4.7.2 seldon-deploy-sdk-1.3.0 six-1.16.0 starlette-0.14.2 typing-extensions-3.10.0.2 urllib3-1.26.6 uvicorn-0.15.0 websocket-client-1.2.1 yarl-1.6.3 zipp-3.5.0[0m
    [35m2021-09-04 08:17:28.855 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:28.855 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m#[0m
    [35m2021-09-04 08:17:28.855 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m# To activate this environment, use[0m
    [35m2021-09-04 08:17:28.855 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m#[0m
    [35m2021-09-04 08:17:28.855 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m#     $ conda activate tempo-bfe7ae29-ca47-4b67-a749-8c99f97d780e[0m
    [35m2021-09-04 08:17:28.855 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m#[0m
    [35m2021-09-04 08:17:28.856 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m# To deactivate an active environment, use[0m
    [35m2021-09-04 08:17:28.856 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m#[0m
    [35m2021-09-04 08:17:28.856 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m#     $ conda deactivate[0m
    [35m2021-09-04 08:17:28.856 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:28.856 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mCollecting packages...[0m
    [35m2021-09-04 08:17:29.562 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mPacking environment at '/home/clive/anaconda3/envs/tempo-bfe7ae29-ca47-4b67-a749-8c99f97d780e' to '/tmp/tmphdhetmmw/environment.tar.gz'[0m
    [########################################] | 100% Completed | 15.9s[0m2m[                                        ] | 0% Completed |  0.0s
    [35m2021-09-04 08:17:45.805 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.805 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m==> WARNING: A newer version of conda exists. <==[0m
    [35m2021-09-04 08:17:45.805 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mcurrent version: 4.7.12[0m
    [35m2021-09-04 08:17:45.806 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mlatest version: 4.10.3[0m
    [35m2021-09-04 08:17:45.806 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.806 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mPlease update conda by running[0m
    [35m2021-09-04 08:17:45.806 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.806 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m$ conda update -n base -c defaults conda[0m
    [35m2021-09-04 08:17:45.806 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.806 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.806 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.953 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m## Package Plan ##[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22menvironment location: /home/clive/anaconda3/envs/tempo-bfe7ae29-ca47-4b67-a749-8c99f97d780e[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mThe following packages will be REMOVED:[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m_libgcc_mutex-0.1-main[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mca-certificates-2021.7.5-h06a4308_1[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mcertifi-2021.5.30-py37h06a4308_0[0m
    [35m2021-09-04 08:17:45.954 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mld_impl_linux-64-2.35.1-h7274673_9[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mlibffi-3.3-he6710b0_2[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mlibgcc-ng-9.1.0-hdf63c60_0[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mlibstdcxx-ng-9.1.0-hdf63c60_0[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mncurses-6.2-he6710b0_1[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mopenssl-1.1.1l-h7f8727e_0[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mpip-21.2.2-py37h06a4308_0[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mpython-3.7.9-h7579374_0[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mreadline-8.1-h27cfd23_0[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22msetuptools-52.0.0-py37h06a4308_0[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22msqlite-3.36.0-hc218d9a_0[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mtk-8.6.10-hbc83047_0[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mwheel-0.37.0-pyhd3eb1b0_0[0m
    [35m2021-09-04 08:17:45.955 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mxz-5.2.5-h7b6447c_0[0m
    [35m2021-09-04 08:17:45.956 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mzlib-1.2.11-h7b6447c_3[0m
    [35m2021-09-04 08:17:45.956 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:45.956 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:17:46.002 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mPreparing transaction: ...working... done[0m
    [35m2021-09-04 08:17:46.151 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mVerifying transaction: ...working... done[0m
    [35m2021-09-04 08:17:46.326 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mExecuting transaction: ...working... done[0m
    [35m2021-09-04 08:18:24.545 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mRemove all packages in environment /home/clive/anaconda3/envs/tempo-bfe7ae29-ca47-4b67-a749-8c99f97d780e:[0m
    [35m2021-09-04 08:18:24.545 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m[0m
    [35m2021-09-04 08:18:24.546 [0m[32m[2/tempo/13 (pid 535666)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 08:18:34.767 [0m[32m[2/tempo/13 (pid 535666)] [0m[22m{'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32), 'output1': 'xgboost prediction'}[0m
    [35m2021-09-04 09:18:37.183 [0m[32m[2/tempo/13 (pid 535666)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 09:18:39.206 [0m[32m[2/end/14 (pid 537519)] [0m[1mTask is starting.[0m
    [35m2021-09-04 09:18:42.875 [0m[32m[2/end/14 (pid 537519)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 09:18:43.323 [0m[1mDone![0m


## Make Predictions with Metaflow Tempo Artifact


```python
from metaflow import Flow
import numpy as np
run = Flow('IrisFlow').latest_run
client = run.data.client_model
client.predict(np.array([[1, 2, 3, 4]]))
```




    {'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32),
     'output1': 'xgboost prediction'}



## Run Flow on AWS and Deploy to Remote Kubernetes

We will now run our flow on AWS Batch and will launch Tempo artifacts onto a remote Kubernetes cluster. 

### Setup AWS Metaflow Support

Note at present this is required even for a local run as artifacts are stored on S3.

[Install Metaflow with remote AWS support](https://docs.metaflow.org/metaflow-on-aws/metaflow-on-aws).

### Seldon Requirements

For deploying to a remote Kubernetes cluster with Seldon Core installed do the following steps:

#### Install Seldon Core on your Kubernetes Cluster

Create a GKE cluster and install Seldon Core on it using [Ansible to install Seldon Core on a Kubernetes cluster](https://github.com/SeldonIO/ansible-k8s-collection).


### K8S Auth from Metaflow

To deploy services to our Kubernetes cluster with Seldon Core installed, Metaflow steps that run on AWS Batch and use tempo will need to be able to access K8S API. This step will depend on whether you're using GKE or AWS EKS to run 
your cluster.

#### Option 1. K8S cluster runs on GKE

We will need to create two files in the flow src folder:

```bash
kubeconfig.yaml
gsa-key.json
```

Follow the steps outlined in [GKE server authentication](https://cloud.google.com/kubernetes-engine/docs/how-to/api-server-authentication#environments-without-gcloud).




#### Option 2. K8S cluster runs on AWS EKS

Make note of two AWS IAM role names, for example find them in the IAM console. The names depend on how you deployed Metaflow and EKS in the first place:

1. The role used by Metaflow tasks executed on AWS Batch. If you used the default CloudFormation template to deploy Metaflow, it is the role that has `*BatchS3TaskRole*` in its name.

2. The role used by EKS nodes. If you used `eksctl` to create your EKS cluster, it is the role that starts with `eksctl-<your-cluster-name>-NodeInstanceRole-*`

Now, we need to make sure that AWS Batch role has permissions to access the K8S cluster. For this, add a policy to the AWS Batch task role(1) that has `eks:*` permissions on your EKS cluster (TODO: narrow this down).

You'll also need to add a mapping for that role to `aws-auth` ConfigMap in `kube-system` namespace. For more details, see [AWS docs](https://docs.aws.amazon.com/eks/latest/userguide/add-user-role.html) (under "To add an IAM user or role to an Amazon EKS cluster"). In short, you'd need to add this to `mapRoles` section in the aws-auth ConfigMap:
```
     - rolearn: <batch task role ARN>
       username: cluster-admin
       groups:
         - system:masters
```

We also need to make sure that the code running in K8S can access S3. For this, add a policy to the EKS node role (2) to allow it to read and write Metaflow S3 buckets.

### S3 Authentication
Services deployed to Seldon will need to access Metaflow S3 bucket to download trained models. The exact configuration will depend on whether you're using GKE or AWS EKS to run your cluster.

From the base templates provided below, create your `k8s/s3_secret.yaml`.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: s3-secret
type: Opaque
stringData:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: aws
  RCLONE_CONFIG_S3_BUCKET_REGION: <region>
  <...cloud-dependent s3 auth settings (see below)>
```

For GKE, to access S3 we'll need to add the following variables to use key/secret auth:
```yaml
  RCLONE_CONFIG_S3_ENV_AUTH: "false"
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: <key>
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: <secret>
```

For AWS EKS, we'll use the instance role assigned to the node, we'll only need to set one env variable:
```yaml
RCLONE_CONFIG_S3_ENV_AUTH: "true"
```

We provide two templates to use in the `k8s` folder:

```
s3_secret.yaml.tmpl.aws
s3_secret.yaml.tmpl.gke
```

Use one to create the file `s3_secret.yaml` in the same folder


## Setup RBAC and Secret on Kubernetes Cluster

These steps assume you have authenticated to your cluster with kubectl configuration


```python
!kubectl create ns production
```

    Error from server (AlreadyExists): namespaces "production" already exists



```python
!kubectl create -f k8s/tempo-pipeline-rbac.yaml -n production
```

    Error from server (AlreadyExists): error when creating "k8s/tempo-pipeline-rbac.yaml": serviceaccounts "tempo-pipeline" already exists
    Error from server (AlreadyExists): error when creating "k8s/tempo-pipeline-rbac.yaml": roles.rbac.authorization.k8s.io "tempo-pipeline" already exists
    Error from server (AlreadyExists): error when creating "k8s/tempo-pipeline-rbac.yaml": rolebindings.rbac.authorization.k8s.io "tempo-pipeline-rolebinding" already exists


Create a Secret from the `k8s/s3_secret.yaml.tmpl` file by adding your AWS Key that can read from S3 and saving as `k8s/s3_secret.yaml`


```python
!kubectl create -f k8s/s3_secret.yaml -n production
```

    Error from server (AlreadyExists): error when creating "k8s/s3_secret.yaml": secrets "s3-secret" already exists


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
    [22mFile persisted at s3://metaflow1-metaflows3bucket-1ou61547mcwbq/metaflow/data/IrisFlow/a94f1cff7702ed70807d16917bb282f51a28511e[K[0m[22m[0m
    [22mIncluding file src/gsa-key.json of size 2KB [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-1ou61547mcwbq/metaflow/data/IrisFlow/c39c6e4ed18f183a2a3ae16b491f2848b57f6824[K[0m[22m[0m
    [22mIncluding file src/kubeconfig.yaml of size 2KB [K[0m[22m[0m
    [22mFile persisted at s3://metaflow1-metaflows3bucket-1ou61547mcwbq/metaflow/data/IrisFlow/a203db53bd80aea425e9fdc206db3d107ce77f69[K[0m[22m[0m
    [35m2021-09-04 10:20:17.745 [0m[1mWorkflow starting (run-id 4):[0m
    [35m2021-09-04 10:20:19.310 [0m[32m[4/start/22 (pid 557187)] [0m[1mTask is starting.[0m
    [35m2021-09-04 09:20:20.457 [0m[32m[4/start/22 (pid 557187)] [0m[22m[ff14310f-71b1-499d-a1c9-40e18f7d1f9a] Task is starting (status SUBMITTED)...[0m
    [35m2021-09-04 09:20:21.568 [0m[32m[4/start/22 (pid 557187)] [0m[22m[ff14310f-71b1-499d-a1c9-40e18f7d1f9a] Task is starting (status STARTING)...[0m
    [35m2021-09-04 09:20:49.878 [0m[32m[4/start/22 (pid 557187)] [0m[22m[ff14310f-71b1-499d-a1c9-40e18f7d1f9a] Task is starting (status RUNNING)...[0m
    [35m2021-09-04 09:20:48.244 [0m[32m[4/start/22 (pid 557187)] [0m[22m[ff14310f-71b1-499d-a1c9-40e18f7d1f9a] Setting up task environment.[0m
    [35m2021-09-04 09:20:54.652 [0m[32m[4/start/22 (pid 557187)] [0m[22m[ff14310f-71b1-499d-a1c9-40e18f7d1f9a] Downloading code package...[0m
    [35m2021-09-04 09:20:55.210 [0m[32m[4/start/22 (pid 557187)] [0m[22m[ff14310f-71b1-499d-a1c9-40e18f7d1f9a] Code package downloaded.[0m
    [35m2021-09-04 09:20:55.222 [0m[32m[4/start/22 (pid 557187)] [0m[22m[ff14310f-71b1-499d-a1c9-40e18f7d1f9a] Task is starting.[0m
    [35m2021-09-04 09:20:55.593 [0m[32m[4/start/22 (pid 557187)] [0m[22m[ff14310f-71b1-499d-a1c9-40e18f7d1f9a] Bootstrapping environment...[0m
    [35m2021-09-04 09:21:19.980 [0m[32m[4/start/22 (pid 557187)] [0m[22m[ff14310f-71b1-499d-a1c9-40e18f7d1f9a] Environment bootstrapped.[0m
    [35m2021-09-04 09:21:30.617 [0m[32m[4/start/22 (pid 557187)] [0m[22m[ff14310f-71b1-499d-a1c9-40e18f7d1f9a] Task finished with exit code 0.[0m
    [35m2021-09-04 10:21:31.495 [0m[32m[4/start/22 (pid 557187)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 10:21:33.227 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[1mTask is starting.[0m
    [35m2021-09-04 10:21:34.342 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[1mTask is starting.[0m
    [35m2021-09-04 09:21:34.342 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Task is starting (status SUBMITTED)...[0m
    [35m2021-09-04 09:21:35.318 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Task is starting (status SUBMITTED)...[0m
    [35m2021-09-04 09:21:36.403 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Task is starting (status RUNNABLE)...[0m
    [35m2021-09-04 09:21:36.410 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Task is starting (status RUNNABLE)...[0m
    [35m2021-09-04 09:21:41.858 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Task is starting (status STARTING)...[0m
    [35m2021-09-04 09:21:41.862 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Task is starting (status STARTING)...[0m
    [35m2021-09-04 09:21:45.123 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Task is starting (status RUNNING)...[0m
    [35m2021-09-04 09:21:45.152 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Task is starting (status RUNNING)...[0m
    [35m2021-09-04 09:21:43.731 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Setting up task environment.[0m
    [35m2021-09-04 09:21:43.729 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Setting up task environment.[0m
    [35m2021-09-04 09:21:50.073 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Downloading code package...[0m
    [35m2021-09-04 09:21:50.637 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Code package downloaded.[0m
    [35m2021-09-04 09:21:50.651 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Task is starting.[0m
    [35m2021-09-04 09:21:51.158 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Bootstrapping environment...[0m
    [35m2021-09-04 09:22:17.930 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Environment bootstrapped.[0m
    [35m2021-09-04 09:22:19.419 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] [09:22:19] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.[0m
    [35m2021-09-04 09:22:27.776 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[22m[7f3ef91d-0952-4520-ae45-938d2e1ea421] Task finished with exit code 0.[0m
    [35m2021-09-04 10:22:29.166 [0m[32m[4/train_xgboost/24 (pid 557741)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 09:21:50.108 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Downloading code package...[0m
    [35m2021-09-04 09:21:50.654 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Code package downloaded.[0m
    [35m2021-09-04 09:21:50.671 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Task is starting.[0m
    [35m2021-09-04 09:21:51.054 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Bootstrapping environment...[0m
    [35m2021-09-04 09:22:17.764 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Environment bootstrapped.[0m
    [35m2021-09-04 09:22:30.355 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[22m[1571eeac-f54e-439c-ac7a-f7142079423c] Task finished with exit code 0.[0m
    [35m2021-09-04 10:22:31.757 [0m[32m[4/train_sklearn/23 (pid 557722)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 10:22:33.297 [0m[32m[4/join/25 (pid 558394)] [0m[1mTask is starting.[0m
    [35m2021-09-04 09:22:34.399 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Task is starting (status SUBMITTED)...[0m
    [35m2021-09-04 09:22:36.635 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Task is starting (status RUNNABLE)...[0m
    [35m2021-09-04 09:22:39.958 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Task is starting (status STARTING)...[0m
    [35m2021-09-04 09:22:42.174 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Task is starting (status RUNNING)...[0m
    [35m2021-09-04 09:22:40.596 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Setting up task environment.[0m
    [35m2021-09-04 09:22:46.988 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Downloading code package...[0m
    [35m2021-09-04 09:22:47.521 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Code package downloaded.[0m
    [35m2021-09-04 09:22:47.532 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Task is starting.[0m
    [35m2021-09-04 09:22:47.906 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Bootstrapping environment...[0m
    [35m2021-09-04 09:23:04.216 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Environment bootstrapped.[0m
    [35m2021-09-04 09:23:12.402 [0m[32m[4/join/25 (pid 558394)] [0m[22m[1b45a3e8-6eec-428e-bf48-9e0ce995b1dc] Task finished with exit code 0.[0m
    [35m2021-09-04 10:23:13.349 [0m[32m[4/join/25 (pid 558394)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 10:23:15.268 [0m[32m[4/tempo/26 (pid 558822)] [0m[1mTask is starting.[0m
    [35m2021-09-04 09:23:16.485 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Task is starting (status SUBMITTED)...[0m
    [35m2021-09-04 09:23:19.778 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Task is starting (status RUNNABLE)...[0m
    [35m2021-09-04 09:23:20.892 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Task is starting (status STARTING)...[0m
    [35m2021-09-04 09:23:23.097 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Task is starting (status RUNNING)...[0m
    [35m2021-09-04 09:23:22.294 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Setting up task environment.[0m
    [35m2021-09-04 09:23:28.662 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Downloading code package...[0m
    [35m2021-09-04 09:23:29.185 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Code package downloaded.[0m
    [35m2021-09-04 09:23:29.196 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Task is starting.[0m
    [35m2021-09-04 09:23:29.574 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Bootstrapping environment...[0m
    [35m2021-09-04 09:23:48.673 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Environment bootstrapped.[0m
    [35m2021-09-04 09:23:49.368 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Pip Test Install: mlops-tempo 0.5.0[0m
    [35m2021-09-04 09:24:07.750 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Pip Test Install: conda_env 2.4.2[0m
    [35m2021-09-04 09:24:07.359 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m
    [35m2021-09-04 09:24:09.714 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] /tmp/tmpyn_akd6x[0m
    [35m2021-09-04 09:24:09.762 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] /tmp/tmpwhkkm565[0m
    [35m2021-09-04 09:24:09.077 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m
    [35m2021-09-04 09:24:09.956 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 09:24:09.956 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 09:24:09.957 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 09:24:10.540 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Warning: you have pip-installed dependencies in your environment file, but you do not list pip itself as one of your conda dependencies.  Conda may not use the correct pip to install your packages, and they may end up in the wrong place.  Please add an explicit pip dependency.  I'm adding one for you, but still nagging you.[0m
    [35m2021-09-04 09:24:10.061 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 09:24:10.061 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 09:24:14.662 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting package metadata (repodata.json): ...working... done[0m
    [35m2021-09-04 09:24:14.928 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Solving environment: ...working... done[0m
    [35m2021-09-04 09:24:15.067 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:15.067 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Downloading and Extracting Packages[0m
    setuptools-52.0.0    | 710 KB    | ########## | 100%[0m58822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] 
    wheel-0.37.0         | 32 KB     | ########## | 100%[0m58822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] 
    openssl-1.1.1l       | 2.5 MB    | ########## | 100%[0m58822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] 
    libgcc-ng-9.3.0      | 4.8 MB    | ########## | 100%[0m58822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] 
    libstdcxx-ng-9.3.0   | 3.1 MB    | ########## | 100%[0m58822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] 
    pip-21.2.2           | 1.8 MB    | ########## | 100%[0m58822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] 
    _openmp_mutex-4.5    | 22 KB     | ########## | 100%[0m58822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] 
    libgomp-9.3.0        | 311 KB    | ########## | 100%[0m58822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] 
    _libgcc_mutex-0.1    | 3 KB      | ########## | 100%[0m58822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] 
    python-3.7.9         | 45.3 MB   | ########## | 100%[0m58822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] 
    [35m2021-09-04 09:24:16.925 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Preparing transaction: ...working... done[0m
    [35m2021-09-04 09:24:17.784 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Verifying transaction: ...working... done[0m
    [35m2021-09-04 09:24:25.024 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Executing transaction: ...working... done[0m
    [35m2021-09-04 09:24:40.467 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Installing pip dependencies: ...working... Ran pip subprocess with arguments:[0m
    [35m2021-09-04 09:24:40.467 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] ['/opt/conda/envs/tempo-bc2369b7-c961-49a5-9c58-ea225c950074/bin/python', '-m', 'pip', 'install', '-U', '-r', '/tmp/condaenv.tof6b1t3.requirements.txt'][0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Pip subprocess output:[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting mlops-tempo[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading mlops_tempo-0.3.0-py3-none-any.whl (74 kB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting mlserver==0.3.2[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading mlserver-0.3.2-py3-none-any.whl (46 kB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting click[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached click-8.0.1-py3-none-any.whl (97 kB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting fastapi[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached fastapi-0.68.1-py3-none-any.whl (52 kB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting orjson[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached orjson-3.6.3-cp37-cp37m-manylinux_2_24_x86_64.whl (234 kB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting uvicorn[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached uvicorn-0.15.0-py3-none-any.whl (54 kB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting numpy[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached numpy-1.21.2-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.7 MB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting grpcio[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached grpcio-1.39.0-cp37-cp37m-manylinux2014_x86_64.whl (4.3 MB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting protobuf[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached protobuf-3.17.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting redis[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached redis-3.5.3-py2.py3-none-any.whl (72 kB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting packaging[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached packaging-21.0-py3-none-any.whl (40 kB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting docker[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached docker-5.0.2-py2.py3-none-any.whl (145 kB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting seldon-deploy-sdk[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached seldon_deploy_sdk-1.3.0-py3-none-any.whl (714 kB)[0m
    [35m2021-09-04 09:24:40.468 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting aiohttp[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached aiohttp-3.7.4.post0-cp37-cp37m-manylinux2014_x86_64.whl (1.3 MB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting requests[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading requests-2.26.0-py2.py3-none-any.whl (62 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting cloudpickle[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached cloudpickle-1.6.0-py3-none-any.whl (23 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting attrs[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached attrs-21.2.0-py2.py3-none-any.whl (53 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting pydantic[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached pydantic-1.8.2-cp37-cp37m-manylinux2014_x86_64.whl (10.1 MB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting python-rclone[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached python_rclone-0.0.2-py3-none-any.whl (4.2 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting conda-pack[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached conda_pack-0.6.0-py2.py3-none-any.whl[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting kubernetes[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached kubernetes-18.20.0-py2.py3-none-any.whl (1.6 MB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting janus[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached janus-0.6.1-py3-none-any.whl (11 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting six>=1.5.2[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting multidict<7.0,>=4.5[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached multidict-5.1.0-cp37-cp37m-manylinux2014_x86_64.whl (142 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting yarl<2.0,>=1.0[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached yarl-1.6.3-cp37-cp37m-manylinux2014_x86_64.whl (294 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting async-timeout<4.0,>=3.0[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached async_timeout-3.0.1-py3-none-any.whl (8.2 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting typing-extensions>=3.6.5[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached typing_extensions-3.10.0.2-py3-none-any.whl (26 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting chardet<5.0,>=2.0[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading chardet-4.0.0-py2.py3-none-any.whl (178 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting idna>=2.0[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading idna-3.2-py3-none-any.whl (59 kB)[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting importlib-metadata[0m
    [35m2021-09-04 09:24:40.469 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading importlib_metadata-4.8.1-py3-none-any.whl (17 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Requirement already satisfied: setuptools in /opt/conda/envs/tempo-bc2369b7-c961-49a5-9c58-ea225c950074/lib/python3.7/site-packages (from conda-pack->mlops-tempo->-r /tmp/condaenv.tof6b1t3.requirements.txt (line 1)) (52.0.0.post20210125)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting websocket-client>=0.32.0[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached websocket_client-1.2.1-py2.py3-none-any.whl (52 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting charset-normalizer~=2.0.0[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading charset_normalizer-2.0.4-py3-none-any.whl (36 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting urllib3<1.27,>=1.21.1[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading urllib3-1.26.6-py2.py3-none-any.whl (138 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/tempo-bc2369b7-c961-49a5-9c58-ea225c950074/lib/python3.7/site-packages (from requests->mlops-tempo->-r /tmp/condaenv.tof6b1t3.requirements.txt (line 1)) (2021.5.30)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting starlette==0.14.2[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached starlette-0.14.2-py3-none-any.whl (60 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting zipp>=0.5[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached zipp-3.5.0-py3-none-any.whl (5.7 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting pyyaml>=5.4.1[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting requests-oauthlib[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting google-auth>=1.0.1[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached google_auth-2.0.2-py2.py3-none-any.whl (152 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting python-dateutil>=2.5.3[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting rsa<5,>=3.1.4[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached rsa-4.7.2-py3-none-any.whl (34 kB)[0m
    [35m2021-09-04 09:24:40.470 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting pyasn1-modules>=0.2.1[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting cachetools<5.0,>=2.0.0[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached cachetools-4.2.2-py3-none-any.whl (11 kB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting pyasn1<0.5.0,>=0.4.6[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting pyparsing>=2.0.2[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting oauthlib>=3.0.0[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached oauthlib-3.1.1-py2.py3-none-any.whl (146 kB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting Authlib<=0.16.0[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached Authlib-0.15.4-py2.py3-none-any.whl (203 kB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting cryptography[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading cryptography-3.4.8-cp36-abi3-manylinux_2_24_x86_64.whl (3.0 MB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting cffi>=1.12[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading cffi-1.14.6-cp37-cp37m-manylinux1_x86_64.whl (402 kB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting pycparser[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Downloading pycparser-2.20-py2.py3-none-any.whl (112 kB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting asgiref>=3.4.0[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached asgiref-3.4.1-py3-none-any.whl (25 kB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting h11>=0.8[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   Using cached h11-0.12.0-py3-none-any.whl (54 kB)[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Installing collected packages: zipp, typing-extensions, pycparser, urllib3, pyasn1, importlib-metadata, idna, charset-normalizer, cffi, starlette, six, rsa, requests, pydantic, pyasn1-modules, oauthlib, multidict, h11, cryptography, click, cachetools, asgiref, yarl, websocket-client, uvicorn, requests-oauthlib, pyyaml, python-dateutil, pyparsing, protobuf, orjson, numpy, grpcio, google-auth, fastapi, chardet, Authlib, attrs, async-timeout, seldon-deploy-sdk, redis, python-rclone, packaging, mlserver, kubernetes, janus, docker, conda-pack, cloudpickle, aiohttp, mlops-tempo[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Successfully installed Authlib-0.15.4 aiohttp-3.7.4.post0 asgiref-3.4.1 async-timeout-3.0.1 attrs-21.2.0 cachetools-4.2.2 cffi-1.14.6 chardet-4.0.0 charset-normalizer-2.0.4 click-8.0.1 cloudpickle-1.6.0 conda-pack-0.6.0 cryptography-3.4.8 docker-5.0.2 fastapi-0.68.1 google-auth-2.0.2 grpcio-1.39.0 h11-0.12.0 idna-3.2 importlib-metadata-4.8.1 janus-0.6.1 kubernetes-18.20.0 mlops-tempo-0.3.0 mlserver-0.3.2 multidict-5.1.0 numpy-1.21.2 oauthlib-3.1.1 orjson-3.6.3 packaging-21.0 protobuf-3.17.3 pyasn1-0.4.8 pyasn1-modules-0.2.8 pycparser-2.20 pydantic-1.8.2 pyparsing-2.4.7 python-dateutil-2.8.2 python-rclone-0.0.2 pyyaml-5.4.1 redis-3.5.3 requests-2.26.0 requests-oauthlib-1.3.0 rsa-4.7.2 seldon-deploy-sdk-1.3.0 six-1.16.0 starlette-0.14.2 typing-extensions-3.10.0.2 urllib3-1.26.6 uvicorn-0.15.0 websocket-client-1.2.1 yarl-1.6.3 zipp-3.5.0[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] done[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] #[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] # To activate this environment, use[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] #[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] #     $ conda activate tempo-bc2369b7-c961-49a5-9c58-ea225c950074[0m
    [35m2021-09-04 09:24:40.471 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] #[0m
    [35m2021-09-04 09:24:40.472 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] # To deactivate an active environment, use[0m
    [35m2021-09-04 09:24:40.472 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] #[0m
    [35m2021-09-04 09:24:40.472 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] #     $ conda deactivate[0m
    [35m2021-09-04 09:24:40.472 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:40.830 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Collecting packages...[0m
    [35m2021-09-04 09:24:41.462 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Packing environment at '/opt/conda/envs/tempo-bc2369b7-c961-49a5-9c58-ea225c950074' to '/tmp/tmpidmn2du9/environment.tar.gz'[0m
    [########################################] | 100% Completed | 10.9s[0m2m[31103662-b002-45a2-bd37-eaf23263f12d] 
    [35m2021-09-04 09:24:52.834 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] ## Package Plan ##[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   environment location: /opt/conda/envs/tempo-bc2369b7-c961-49a5-9c58-ea225c950074[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] The following packages will be REMOVED:[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   _libgcc_mutex-0.1-main[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   _openmp_mutex-4.5-1_gnu[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   ca-certificates-2021.7.5-h06a4308_1[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   certifi-2021.5.30-py37h06a4308_0[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   ld_impl_linux-64-2.35.1-h7274673_9[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   libffi-3.3-he6710b0_2[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   libgcc-ng-9.3.0-h5101ec6_17[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   libgomp-9.3.0-h5101ec6_17[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   libstdcxx-ng-9.3.0-hd4cf53a_17[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   ncurses-6.2-he6710b0_1[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   openssl-1.1.1l-h7f8727e_0[0m
    [35m2021-09-04 09:24:52.835 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   pip-21.2.2-py37h06a4308_0[0m
    [35m2021-09-04 09:24:52.836 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   python-3.7.9-h7579374_0[0m
    [35m2021-09-04 09:24:52.836 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   readline-8.1-h27cfd23_0[0m
    [35m2021-09-04 09:24:52.836 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   setuptools-52.0.0-py37h06a4308_0[0m
    [35m2021-09-04 09:24:52.836 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   sqlite-3.36.0-hc218d9a_0[0m
    [35m2021-09-04 09:24:52.836 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   tk-8.6.10-hbc83047_0[0m
    [35m2021-09-04 09:24:52.836 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   wheel-0.37.0-pyhd3eb1b0_0[0m
    [35m2021-09-04 09:24:52.836 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   xz-5.2.5-h7b6447c_0[0m
    [35m2021-09-04 09:24:52.836 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d]   zlib-1.2.11-h7b6447c_3[0m
    [35m2021-09-04 09:24:52.836 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:52.836 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:52.869 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Preparing transaction: ...working... done[0m
    [35m2021-09-04 09:24:52.971 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Verifying transaction: ...working... done[0m
    [35m2021-09-04 09:24:53.160 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Executing transaction: ...working... done[0m
    [35m2021-09-04 09:24:52.732 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:24:52.732 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Remove all packages in environment /opt/conda/envs/tempo-bc2369b7-c961-49a5-9c58-ea225c950074:[0m
    [35m2021-09-04 09:24:52.732 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d][0m
    [35m2021-09-04 09:26:23.089 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-09-04 09:26:33.373 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] {'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32), 'output1': 'xgboost prediction'}[0m
    [35m2021-09-04 09:26:37.309 [0m[32m[4/tempo/26 (pid 558822)] [0m[22m[31103662-b002-45a2-bd37-eaf23263f12d] Task finished with exit code 0.[0m
    [35m2021-09-04 10:26:38.179 [0m[32m[4/tempo/26 (pid 558822)] [0m[1mTask finished successfully.[0m
    [35m2021-09-04 10:26:39.233 [0m[32m[4/end/27 (pid 560413)] [0m[1mTask is starting.[0m
    [35m2021-09-04 09:26:40.258 [0m[32m[4/end/27 (pid 560413)] [0m[22m[a0ec6904-72ac-47bb-8664-ebf7f245eeba] Task is starting (status SUBMITTED)...[0m
    [35m2021-09-04 09:26:45.703 [0m[32m[4/end/27 (pid 560413)] [0m[22m[a0ec6904-72ac-47bb-8664-ebf7f245eeba] Task is starting (status STARTING)...[0m
    [35m2021-09-04 09:26:48.963 [0m[32m[4/end/27 (pid 560413)] [0m[22m[a0ec6904-72ac-47bb-8664-ebf7f245eeba] Task is starting (status RUNNING)...[0m


## Make Predictions with Metaflow Tempo Artifact


```python
from metaflow import Flow
run = Flow('IrisFlow').latest_run
client = run.data.client_model
import numpy as np
client.predict(np.array([[1, 2, 3, 4]]))
```




    {'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32),
     'output1': 'xgboost prediction'}


