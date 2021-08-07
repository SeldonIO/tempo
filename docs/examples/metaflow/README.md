# Metaflow with Tempo Example

We will train two models and deploy them with tempo within a Metaflow pipeline. To understand the core example see [here](https://tempo.readthedocs.io/en/latest/examples/multi-model/README.html)

![archtecture](architecture.png)

## MetaFlow Prequisites

Install metaflow

```
pip install metaflow
```

[Install Metaflow with remote AWS support](https://docs.metaflow.org/metaflow-on-aws/metaflow-on-aws).




## Tempo Requirements

For deploy to a remote Kubernetes cluster:

### GKE Cluster

Create a GKE cluster and install Seldon Core on it using [Ansible to install Seldon Core on a Kubernetes cluster](https://github.com/SeldonIO/ansible-k8s-collection).

For GKE we will need to create two files in the flow src folder:

```bash
kubeconfig.yaml
gsa-key.json
```

Follow the steps outlined in [GKE server authentication](https://cloud.google.com/kubernetes-engine/docs/how-to/api-server-authentication#environments-without-gcloud).




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
    [35m2021-08-07 19:15:22.217 [0m[1mWorkflow starting (run-id 27):[0m
    [35m2021-08-07 19:15:23.659 [0m[32m[27/start/151 (pid 2044924)] [0m[1mTask is starting.[0m
    [35m2021-08-07 19:15:27.443 [0m[32m[27/start/151 (pid 2044924)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:15:29.009 [0m[32m[27/train_sklearn/152 (pid 2045008)] [0m[1mTask is starting.[0m
    [35m2021-08-07 19:15:30.083 [0m[32m[27/train_xgboost/153 (pid 2045034)] [0m[1mTask is starting.[0m
    [35m2021-08-07 18:15:32.243 [0m[32m[27/train_xgboost/153 (pid 2045034)] [0m[22m[19:15:32] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.[0m
    [35m2021-08-07 19:15:33.033 [0m[32m[27/train_sklearn/152 (pid 2045008)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:15:34.239 [0m[32m[27/train_xgboost/153 (pid 2045034)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:15:35.924 [0m[32m[27/join/154 (pid 2045162)] [0m[1mTask is starting.[0m
    [35m2021-08-07 19:15:41.508 [0m[32m[27/join/154 (pid 2045162)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:15:43.383 [0m[32m[27/tempo/155 (pid 2045276)] [0m[1mTask is starting.[0m
    [35m2021-08-07 18:15:45.259 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mPip Test Install: mlops-tempo 0.4.0.dev3[0m
    [35m2021-08-07 18:15:46.201 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mPip Test Install: conda_env 2.4.2[0m
    [35m2021-08-07 18:15:47.491 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m/tmp/tmpq6lwdtfi[0m
    [35m2021-08-07 18:15:47.622 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m/tmp/tmpnpbmgvde[0m
    [35m2021-08-07 18:15:48.325 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:15:48.326 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:15:48.326 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:15:48.698 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:15:48.698 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:15:49.783 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mWarning: you have pip-installed dependencies in your environment file, but you do not list pip itself as one of your conda dependencies.  Conda may not use the correct pip to install your packages, and they may end up in the wrong place.  Please add an explicit pip dependency.  I'm adding one for you, but still nagging you.[0m
    [35m2021-08-07 18:15:53.367 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting package metadata (repodata.json): ...working... done[0m
    [35m2021-08-07 18:15:54.765 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mSolving environment: ...working... done[0m
    [35m2021-08-07 18:15:54.876 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:15:55.237 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mPreparing transaction: ...working... done[0m
    [35m2021-08-07 18:15:55.902 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mVerifying transaction: ...working... done[0m
    [35m2021-08-07 18:15:56.601 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mExecuting transaction: ...working... done[0m
    [35m2021-08-07 18:16:14.441 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mRan pip subprocess with arguments:[0m
    [35m2021-08-07 18:16:14.441 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m['/home/clive/anaconda3/envs/tempo-bfa84d96-4681-4fef-bd43-4911a7b8ff79/bin/python', '-m', 'pip', 'install', '-U', '-r', '/tmp/condaenv.hywlezpj.requirements.txt'][0m
    [35m2021-08-07 18:16:14.441 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mPip subprocess output:[0m
    [35m2021-08-07 18:16:14.441 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting mlops-tempo[0m
    [35m2021-08-07 18:16:14.441 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached mlops_tempo-0.3.0-py3-none-any.whl (74 kB)[0m
    [35m2021-08-07 18:16:14.441 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting mlserver==0.3.2[0m
    [35m2021-08-07 18:16:14.441 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached mlserver-0.3.2-py3-none-any.whl (46 kB)[0m
    [35m2021-08-07 18:16:14.441 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting orjson[0m
    [35m2021-08-07 18:16:14.441 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached orjson-3.6.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (233 kB)[0m
    [35m2021-08-07 18:16:14.441 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting uvicorn[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached uvicorn-0.14.0-py3-none-any.whl (50 kB)[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting protobuf[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached protobuf-3.17.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting grpcio[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached grpcio-1.39.0-cp37-cp37m-manylinux2014_x86_64.whl (4.3 MB)[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting fastapi[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached fastapi-0.68.0-py3-none-any.whl (52 kB)[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting click[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached click-8.0.1-py3-none-any.whl (97 kB)[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting numpy[0m
    [35m2021-08-07 18:16:14.442 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached numpy-1.21.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.7 MB)[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting docker[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached docker-5.0.0-py2.py3-none-any.whl (146 kB)[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting python-rclone[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached python_rclone-0.0.2-py3-none-any.whl (4.2 kB)[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting pydantic[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached pydantic-1.8.2-cp37-cp37m-manylinux2014_x86_64.whl (10.1 MB)[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting conda-pack[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached conda_pack-0.6.0-py2.py3-none-any.whl[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting janus[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached janus-0.6.1-py3-none-any.whl (11 kB)[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting redis[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached redis-3.5.3-py2.py3-none-any.whl (72 kB)[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting kubernetes[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached kubernetes-17.17.0-py3-none-any.whl (1.8 MB)[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting requests[0m
    [35m2021-08-07 18:16:14.443 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached requests-2.26.0-py2.py3-none-any.whl (62 kB)[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting cloudpickle[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached cloudpickle-1.6.0-py3-none-any.whl (23 kB)[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting packaging[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached packaging-21.0-py3-none-any.whl (40 kB)[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting aiohttp[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached aiohttp-3.7.4.post0-cp37-cp37m-manylinux2014_x86_64.whl (1.3 MB)[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting attrs[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached attrs-21.2.0-py2.py3-none-any.whl (53 kB)[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting seldon-deploy-sdk[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached seldon_deploy_sdk-1.3.0-py3-none-any.whl (714 kB)[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting six>=1.5.2[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached six-1.16.0-py2.py3-none-any.whl (11 kB)[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting multidict<7.0,>=4.5[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached multidict-5.1.0-cp37-cp37m-manylinux2014_x86_64.whl (142 kB)[0m
    [35m2021-08-07 18:16:14.444 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting chardet<5.0,>=2.0[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached chardet-4.0.0-py2.py3-none-any.whl (178 kB)[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting yarl<2.0,>=1.0[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached yarl-1.6.3-cp37-cp37m-manylinux2014_x86_64.whl (294 kB)[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting typing-extensions>=3.6.5[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached typing_extensions-3.10.0.0-py3-none-any.whl (26 kB)[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting async-timeout<4.0,>=3.0[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached async_timeout-3.0.1-py3-none-any.whl (8.2 kB)[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting idna>=2.0[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached idna-3.2-py3-none-any.whl (59 kB)[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting importlib-metadata[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached importlib_metadata-4.6.3-py3-none-any.whl (17 kB)[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mRequirement already satisfied: setuptools in /home/clive/anaconda3/envs/tempo-bfa84d96-4681-4fef-bd43-4911a7b8ff79/lib/python3.7/site-packages (from conda-pack->mlops-tempo->-r /tmp/condaenv.hywlezpj.requirements.txt (line 1)) (52.0.0.post20210125)[0m
    [35m2021-08-07 18:16:14.445 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting websocket-client>=0.32.0[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached websocket_client-1.1.1-py2.py3-none-any.whl (68 kB)[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mRequirement already satisfied: certifi>=2017.4.17 in /home/clive/anaconda3/envs/tempo-bfa84d96-4681-4fef-bd43-4911a7b8ff79/lib/python3.7/site-packages (from requests->mlops-tempo->-r /tmp/condaenv.hywlezpj.requirements.txt (line 1)) (2021.5.30)[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting urllib3<1.27,>=1.21.1[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached urllib3-1.26.6-py2.py3-none-any.whl (138 kB)[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting charset-normalizer~=2.0.0[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached charset_normalizer-2.0.4-py3-none-any.whl (36 kB)[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting starlette==0.14.2[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached starlette-0.14.2-py3-none-any.whl (60 kB)[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting zipp>=0.5[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached zipp-3.5.0-py3-none-any.whl (5.7 kB)[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting pyyaml>=3.12[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting google-auth>=1.0.1[0m
    [35m2021-08-07 18:16:14.446 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached google_auth-1.34.0-py2.py3-none-any.whl (152 kB)[0m
    [35m2021-08-07 18:16:14.447 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting python-dateutil>=2.5.3[0m
    [35m2021-08-07 18:16:14.447 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)[0m
    [35m2021-08-07 18:16:14.447 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting requests-oauthlib[0m
    [35m2021-08-07 18:16:16.954 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)[0m
    [35m2021-08-07 18:16:16.954 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting cachetools<5.0,>=2.0.0[0m
    [35m2021-08-07 18:16:16.954 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached cachetools-4.2.2-py3-none-any.whl (11 kB)[0m
    [35m2021-08-07 18:16:16.954 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting rsa<5,>=3.1.4[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached rsa-4.7.2-py3-none-any.whl (34 kB)[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting pyasn1-modules>=0.2.1[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting pyasn1<0.5.0,>=0.4.6[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting pyparsing>=2.0.2[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting oauthlib>=3.0.0[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached oauthlib-3.1.1-py2.py3-none-any.whl (146 kB)[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting Authlib<=0.16.0[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached Authlib-0.15.4-py2.py3-none-any.whl (203 kB)[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting cryptography[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached cryptography-3.4.7-cp36-abi3-manylinux2014_x86_64.whl (3.2 MB)[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting cffi>=1.12[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached cffi-1.14.6-cp37-cp37m-manylinux1_x86_64.whl (402 kB)[0m
    [35m2021-08-07 18:16:16.955 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting pycparser[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached pycparser-2.20-py2.py3-none-any.whl (112 kB)[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting h11>=0.8[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached h11-0.12.0-py3-none-any.whl (54 kB)[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting asgiref>=3.3.4[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mUsing cached asgiref-3.4.1-py3-none-any.whl (25 kB)[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mInstalling collected packages: zipp, typing-extensions, pycparser, urllib3, pyasn1, importlib-metadata, idna, charset-normalizer, cffi, starlette, six, rsa, requests, pydantic, pyasn1-modules, oauthlib, multidict, h11, cryptography, click, cachetools, asgiref, yarl, websocket-client, uvicorn, requests-oauthlib, pyyaml, python-dateutil, pyparsing, protobuf, orjson, numpy, grpcio, google-auth, fastapi, chardet, Authlib, attrs, async-timeout, seldon-deploy-sdk, redis, python-rclone, packaging, mlserver, kubernetes, janus, docker, conda-pack, cloudpickle, aiohttp, mlops-tempo[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mSuccessfully installed Authlib-0.15.4 aiohttp-3.7.4.post0 asgiref-3.4.1 async-timeout-3.0.1 attrs-21.2.0 cachetools-4.2.2 cffi-1.14.6 chardet-4.0.0 charset-normalizer-2.0.4 click-8.0.1 cloudpickle-1.6.0 conda-pack-0.6.0 cryptography-3.4.7 docker-5.0.0 fastapi-0.68.0 google-auth-1.34.0 grpcio-1.39.0 h11-0.12.0 idna-3.2 importlib-metadata-4.6.3 janus-0.6.1 kubernetes-17.17.0 mlops-tempo-0.3.0 mlserver-0.3.2 multidict-5.1.0 numpy-1.21.1 oauthlib-3.1.1 orjson-3.6.1 packaging-21.0 protobuf-3.17.3 pyasn1-0.4.8 pyasn1-modules-0.2.8 pycparser-2.20 pydantic-1.8.2 pyparsing-2.4.7 python-dateutil-2.8.2 python-rclone-0.0.2 pyyaml-5.4.1 redis-3.5.3 requests-2.26.0 requests-oauthlib-1.3.0 rsa-4.7.2 seldon-deploy-sdk-1.3.0 six-1.16.0 starlette-0.14.2 typing-extensions-3.10.0.0 urllib3-1.26.6 uvicorn-0.14.0 websocket-client-1.1.1 yarl-1.6.3 zipp-3.5.0[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m#[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m# To activate this environment, use[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m#[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m#     $ conda activate tempo-bfa84d96-4681-4fef-bd43-4911a7b8ff79[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m#[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m# To deactivate an active environment, use[0m
    [35m2021-08-07 18:16:16.956 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m#[0m
    [35m2021-08-07 18:16:16.957 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m#     $ conda deactivate[0m
    [35m2021-08-07 18:16:16.957 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:16.957 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mCollecting packages...[0m
    [35m2021-08-07 18:16:17.537 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mPacking environment at '/home/clive/anaconda3/envs/tempo-bfa84d96-4681-4fef-bd43-4911a7b8ff79' to '/tmp/tmpwmel5bgf/environment.tar.gz'[0m
    [########################################] | 100% Completed | 12.0s[0m[22m[                                        ] | 0% Completed |  0.0s
    [35m2021-08-07 18:16:29.843 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.843 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m==> WARNING: A newer version of conda exists. <==[0m
    [35m2021-08-07 18:16:29.843 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mcurrent version: 4.7.12[0m
    [35m2021-08-07 18:16:29.843 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mlatest version: 4.10.3[0m
    [35m2021-08-07 18:16:29.843 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.843 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mPlease update conda by running[0m
    [35m2021-08-07 18:16:29.843 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.843 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m$ conda update -n base -c defaults conda[0m
    [35m2021-08-07 18:16:29.843 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.844 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.844 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m## Package Plan ##[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22menvironment location: /home/clive/anaconda3/envs/tempo-bfa84d96-4681-4fef-bd43-4911a7b8ff79[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mThe following packages will be REMOVED:[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m_libgcc_mutex-0.1-main[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mca-certificates-2021.7.5-h06a4308_1[0m
    [35m2021-08-07 18:16:29.968 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mcertifi-2021.5.30-py37h06a4308_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mld_impl_linux-64-2.35.1-h7274673_9[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mlibffi-3.3-he6710b0_2[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mlibgcc-ng-9.1.0-hdf63c60_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mlibstdcxx-ng-9.1.0-hdf63c60_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mncurses-6.2-he6710b0_1[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mopenssl-1.1.1k-h27cfd23_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mpip-21.2.2-py37h06a4308_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mpython-3.7.9-h7579374_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mreadline-8.1-h27cfd23_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22msetuptools-52.0.0-py37h06a4308_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22msqlite-3.36.0-hc218d9a_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mtk-8.6.10-hbc83047_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mwheel-0.36.2-pyhd3eb1b0_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mxz-5.2.5-h7b6447c_0[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mzlib-1.2.11-h7b6447c_3[0m
    [35m2021-08-07 18:16:29.969 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:29.970 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:16:30.007 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mPreparing transaction: ...working... done[0m
    [35m2021-08-07 18:16:30.121 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mVerifying transaction: ...working... done[0m
    [35m2021-08-07 18:16:30.261 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mExecuting transaction: ...working... done[0m
    [35m2021-08-07 18:17:08.886 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mRemove all packages in environment /home/clive/anaconda3/envs/tempo-bfa84d96-4681-4fef-bd43-4911a7b8ff79:[0m
    [35m2021-08-07 18:17:08.887 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m[0m
    [35m2021-08-07 18:17:08.887 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22mInsights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:17:18.985 [0m[32m[27/tempo/155 (pid 2045276)] [0m[22m{'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32), 'output1': 'xgboost prediction'}[0m
    [35m2021-08-07 19:17:21.262 [0m[32m[27/tempo/155 (pid 2045276)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:17:23.169 [0m[32m[27/end/156 (pid 2047464)] [0m[1mTask is starting.[0m
    [35m2021-08-07 19:17:26.746 [0m[32m[27/end/156 (pid 2047464)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:17:27.207 [0m[1mDone![0m


Use the saved client from the Flow to make predictions


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



## Run Flow on AWS and Deploy to Kubernetes


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
    [35m2021-08-07 19:25:55.509 [0m[1mWorkflow starting (run-id 28):[0m
    [35m2021-08-07 19:25:57.115 [0m[32m[28/start/158 (pid 2050712)] [0m[1mTask is starting.[0m
    [35m2021-08-07 18:25:58.583 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-07 18:26:00.689 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-07 18:26:05.040 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Task is starting (status STARTING)...[0m
    [35m2021-08-07 18:26:32.356 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Task is starting (status RUNNING)...[0m
    [35m2021-08-07 18:26:30.757 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Setting up task environment.[0m
    [35m2021-08-07 18:26:36.904 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Downloading code package...[0m
    [35m2021-08-07 18:26:37.441 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Code package downloaded.[0m
    [35m2021-08-07 18:26:37.452 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Task is starting.[0m
    [35m2021-08-07 18:26:37.815 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Bootstrapping environment...[0m
    [35m2021-08-07 18:27:02.311 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Environment bootstrapped.[0m
    [35m2021-08-07 18:27:12.967 [0m[32m[28/start/158 (pid 2050712)] [0m[22m[dc35448a-1263-4847-b5a3-1f7c8ce72e47] Task finished with exit code 0.[0m
    [35m2021-08-07 19:27:13.844 [0m[32m[28/start/158 (pid 2050712)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:27:15.421 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[1mTask is starting.[0m
    [35m2021-08-07 19:27:16.479 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[1mTask is starting.[0m
    [35m2021-08-07 18:27:16.480 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-07 18:27:17.415 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-07 18:27:21.843 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-07 18:27:21.850 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-07 18:27:24.016 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Task is starting (status STARTING)...[0m
    [35m2021-08-07 18:27:24.036 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Task is starting (status STARTING)...[0m
    [35m2021-08-07 18:27:25.109 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Task is starting (status RUNNING)...[0m
    [35m2021-08-07 18:27:26.205 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Task is starting (status RUNNING)...[0m
    [35m2021-08-07 18:27:24.582 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Setting up task environment.[0m
    [35m2021-08-07 18:27:30.807 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Downloading code package...[0m
    [35m2021-08-07 18:27:24.595 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Setting up task environment.[0m
    [35m2021-08-07 18:27:30.833 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Downloading code package...[0m
    [35m2021-08-07 18:27:31.357 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Code package downloaded.[0m
    [35m2021-08-07 18:27:31.369 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Task is starting.[0m
    [35m2021-08-07 18:27:31.945 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Bootstrapping environment...[0m
    [35m2021-08-07 18:27:58.337 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Environment bootstrapped.[0m
    [35m2021-08-07 18:27:59.710 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] [18:27:59] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.[0m
    [35m2021-08-07 18:28:09.638 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[22m[c028d2e9-c452-4a42-a285-06dcadc614db] Task finished with exit code 0.[0m
    [35m2021-08-07 19:28:10.457 [0m[32m[28/train_xgboost/160 (pid 2051273)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 18:27:31.359 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Code package downloaded.[0m
    [35m2021-08-07 18:27:31.371 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Task is starting.[0m
    [35m2021-08-07 18:27:31.926 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Bootstrapping environment...[0m
    [35m2021-08-07 18:27:57.661 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Environment bootstrapped.[0m
    [35m2021-08-07 18:28:10.994 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[22m[49bffe73-d5b1-493c-8272-25df14e090f7] Task finished with exit code 0.[0m
    [35m2021-08-07 19:28:11.533 [0m[32m[28/train_sklearn/159 (pid 2051254)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:28:12.541 [0m[32m[28/join/161 (pid 2051623)] [0m[1mTask is starting.[0m
    [35m2021-08-07 18:28:13.576 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-07 18:28:14.662 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-07 18:28:16.833 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Task is starting (status STARTING)...[0m
    [35m2021-08-07 18:28:21.190 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Task is starting (status RUNNING)...[0m
    [35m2021-08-07 18:28:19.106 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Setting up task environment.[0m
    [35m2021-08-07 18:28:25.194 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Downloading code package...[0m
    [35m2021-08-07 18:28:25.705 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Code package downloaded.[0m
    [35m2021-08-07 18:28:25.717 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Task is starting.[0m
    [35m2021-08-07 18:28:26.084 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Bootstrapping environment...[0m
    [35m2021-08-07 18:28:42.377 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Environment bootstrapped.[0m
    [35m2021-08-07 18:28:51.036 [0m[32m[28/join/161 (pid 2051623)] [0m[22m[624716f9-9250-4627-9e46-d62a71cb1c41] Task finished with exit code 0.[0m
    [35m2021-08-07 19:28:51.922 [0m[32m[28/join/161 (pid 2051623)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:28:53.763 [0m[32m[28/tempo/162 (pid 2051921)] [0m[1mTask is starting.[0m
    [35m2021-08-07 18:28:54.749 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-07 18:28:58.032 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-07 18:28:59.120 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Task is starting (status STARTING)...[0m
    [35m2021-08-07 18:29:01.310 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Task is starting (status RUNNING)...[0m
    [35m2021-08-07 18:29:00.084 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Setting up task environment.[0m
    [35m2021-08-07 18:29:06.238 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Downloading code package...[0m
    [35m2021-08-07 18:29:06.744 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Code package downloaded.[0m
    [35m2021-08-07 18:29:06.757 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Task is starting.[0m
    [35m2021-08-07 18:29:07.121 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Bootstrapping environment...[0m
    [35m2021-08-07 18:29:26.172 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Environment bootstrapped.[0m
    [35m2021-08-07 18:29:26.825 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Pip Test Install: mlops-tempo 0.4.0.dev3[0m
    [35m2021-08-07 18:29:44.497 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Pip Test Install: conda_env 2.4.2[0m
    [35m2021-08-07 18:29:44.207 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m
    [35m2021-08-07 18:29:45.810 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m
    [35m2021-08-07 18:29:46.501 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] /tmp/tmphf2iqzfe[0m
    [35m2021-08-07 18:29:46.543 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] /tmp/tmprpc_yt0_[0m
    [35m2021-08-07 18:29:47.329 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Warning: you have pip-installed dependencies in your environment file, but you do not list pip itself as one of your conda dependencies.  Conda may not use the correct pip to install your packages, and they may end up in the wrong place.  Please add an explicit pip dependency.  I'm adding one for you, but still nagging you.[0m
    [35m2021-08-07 18:29:46.733 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:29:46.733 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:29:46.734 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:29:46.848 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:29:46.848 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:29:51.259 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting package metadata (repodata.json): ...working... done[0m
    [35m2021-08-07 18:29:51.369 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Solving environment: ...working... done[0m
    [35m2021-08-07 18:29:51.500 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:29:51.500 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Downloading and Extracting Packages[0m
    setuptools-52.0.0    | 710 KB    | ########## | 100%[0m 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] 
    pip-21.2.2           | 1.8 MB    | ########## | 100%[0m 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] 
    [35m2021-08-07 18:29:51.470 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:29:51.471 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:29:51.471 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] ==> WARNING: A newer version of conda exists. <==[0m
    [35m2021-08-07 18:29:51.471 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   current version: 4.7.12[0m
    [35m2021-08-07 18:29:51.471 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   latest version: 4.10.3[0m
    [35m2021-08-07 18:29:51.471 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:29:51.471 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Please update conda by running[0m
    [35m2021-08-07 18:29:51.471 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:29:51.471 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]     $ conda update -n base -c defaults conda[0m
    [35m2021-08-07 18:29:51.471 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:29:51.471 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    python-3.7.9         | 45.3 MB   | ########## | 100%[0m 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] 
    wheel-0.36.2         | 33 KB     | ########## | 100%[0m 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] 
    _libgcc_mutex-0.1    | 3 KB      | ########## | 100%[0m 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] 
    [35m2021-08-07 18:29:53.003 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Preparing transaction: ...working... done[0m
    [35m2021-08-07 18:29:53.852 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Verifying transaction: ...working... done[0m
    [35m2021-08-07 18:30:00.594 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Executing transaction: ...working... done[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Ran pip subprocess with arguments:[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] ['/opt/conda/envs/tempo-a1d8e8bb-4b2e-4017-8b60-a5bd0a95a426/bin/python', '-m', 'pip', 'install', '-U', '-r', '/tmp/condaenv.zftueo0g.requirements.txt'][0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Pip subprocess output:[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting mlops-tempo[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading mlops_tempo-0.3.0-py3-none-any.whl (74 kB)[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting mlserver==0.3.2[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading mlserver-0.3.2-py3-none-any.whl (46 kB)[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting uvicorn[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached uvicorn-0.14.0-py3-none-any.whl (50 kB)[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting grpcio[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached grpcio-1.39.0-cp37-cp37m-manylinux2014_x86_64.whl (4.3 MB)[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting numpy[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached numpy-1.21.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.7 MB)[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting protobuf[0m
    [35m2021-08-07 18:30:16.091 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached protobuf-3.17.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting click[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached click-8.0.1-py3-none-any.whl (97 kB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting orjson[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached orjson-3.6.1-cp37-cp37m-manylinux_2_24_x86_64.whl (233 kB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting fastapi[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached fastapi-0.68.0-py3-none-any.whl (52 kB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting pydantic[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached pydantic-1.8.2-cp37-cp37m-manylinux2014_x86_64.whl (10.1 MB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting python-rclone[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached python_rclone-0.0.2-py3-none-any.whl (4.2 kB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting janus[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached janus-0.6.1-py3-none-any.whl (11 kB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting redis[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached redis-3.5.3-py2.py3-none-any.whl (72 kB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting requests[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading requests-2.26.0-py2.py3-none-any.whl (62 kB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting aiohttp[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached aiohttp-3.7.4.post0-cp37-cp37m-manylinux2014_x86_64.whl (1.3 MB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting seldon-deploy-sdk[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached seldon_deploy_sdk-1.3.0-py3-none-any.whl (714 kB)[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting attrs[0m
    [35m2021-08-07 18:30:16.092 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached attrs-21.2.0-py2.py3-none-any.whl (53 kB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting conda-pack[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached conda_pack-0.6.0-py2.py3-none-any.whl[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting docker[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached docker-5.0.0-py2.py3-none-any.whl (146 kB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting cloudpickle[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached cloudpickle-1.6.0-py3-none-any.whl (23 kB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting kubernetes[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached kubernetes-17.17.0-py3-none-any.whl (1.8 MB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting packaging[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached packaging-21.0-py3-none-any.whl (40 kB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting six>=1.5.2[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting multidict<7.0,>=4.5[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached multidict-5.1.0-cp37-cp37m-manylinux2014_x86_64.whl (142 kB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting async-timeout<4.0,>=3.0[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached async_timeout-3.0.1-py3-none-any.whl (8.2 kB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting typing-extensions>=3.6.5[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached typing_extensions-3.10.0.0-py3-none-any.whl (26 kB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting yarl<2.0,>=1.0[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached yarl-1.6.3-cp37-cp37m-manylinux2014_x86_64.whl (294 kB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting chardet<5.0,>=2.0[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading chardet-4.0.0-py2.py3-none-any.whl (178 kB)[0m
    [35m2021-08-07 18:30:16.093 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting idna>=2.0[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading idna-3.2-py3-none-any.whl (59 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting importlib-metadata[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached importlib_metadata-4.6.3-py3-none-any.whl (17 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Requirement already satisfied: setuptools in /opt/conda/envs/tempo-a1d8e8bb-4b2e-4017-8b60-a5bd0a95a426/lib/python3.7/site-packages (from conda-pack->mlops-tempo->-r /tmp/condaenv.zftueo0g.requirements.txt (line 1)) (52.0.0.post20210125)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting websocket-client>=0.32.0[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached websocket_client-1.1.1-py2.py3-none-any.whl (68 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/tempo-a1d8e8bb-4b2e-4017-8b60-a5bd0a95a426/lib/python3.7/site-packages (from requests->mlops-tempo->-r /tmp/condaenv.zftueo0g.requirements.txt (line 1)) (2021.5.30)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting charset-normalizer~=2.0.0[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading charset_normalizer-2.0.4-py3-none-any.whl (36 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting urllib3<1.27,>=1.21.1[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading urllib3-1.26.6-py2.py3-none-any.whl (138 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting starlette==0.14.2[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached starlette-0.14.2-py3-none-any.whl (60 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting zipp>=0.5[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached zipp-3.5.0-py3-none-any.whl (5.7 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting requests-oauthlib[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting python-dateutil>=2.5.3[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting google-auth>=1.0.1[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached google_auth-1.34.0-py2.py3-none-any.whl (152 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting pyyaml>=3.12[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)[0m
    [35m2021-08-07 18:30:16.094 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting cachetools<5.0,>=2.0.0[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached cachetools-4.2.2-py3-none-any.whl (11 kB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting rsa<5,>=3.1.4[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached rsa-4.7.2-py3-none-any.whl (34 kB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting pyasn1-modules>=0.2.1[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting pyasn1<0.5.0,>=0.4.6[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting pyparsing>=2.0.2[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting oauthlib>=3.0.0[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached oauthlib-3.1.1-py2.py3-none-any.whl (146 kB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting Authlib<=0.16.0[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached Authlib-0.15.4-py2.py3-none-any.whl (203 kB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting cryptography[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading cryptography-3.4.7-cp36-abi3-manylinux2014_x86_64.whl (3.2 MB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting cffi>=1.12[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading cffi-1.14.6-cp37-cp37m-manylinux1_x86_64.whl (402 kB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting pycparser[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Downloading pycparser-2.20-py2.py3-none-any.whl (112 kB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting h11>=0.8[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached h11-0.12.0-py3-none-any.whl (54 kB)[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting asgiref>=3.3.4[0m
    [35m2021-08-07 18:30:16.095 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   Using cached asgiref-3.4.1-py3-none-any.whl (25 kB)[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Installing collected packages: zipp, typing-extensions, pycparser, urllib3, pyasn1, importlib-metadata, idna, charset-normalizer, cffi, starlette, six, rsa, requests, pydantic, pyasn1-modules, oauthlib, multidict, h11, cryptography, click, cachetools, asgiref, yarl, websocket-client, uvicorn, requests-oauthlib, pyyaml, python-dateutil, pyparsing, protobuf, orjson, numpy, grpcio, google-auth, fastapi, chardet, Authlib, attrs, async-timeout, seldon-deploy-sdk, redis, python-rclone, packaging, mlserver, kubernetes, janus, docker, conda-pack, cloudpickle, aiohttp, mlops-tempo[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Successfully installed Authlib-0.15.4 aiohttp-3.7.4.post0 asgiref-3.4.1 async-timeout-3.0.1 attrs-21.2.0 cachetools-4.2.2 cffi-1.14.6 chardet-4.0.0 charset-normalizer-2.0.4 click-8.0.1 cloudpickle-1.6.0 conda-pack-0.6.0 cryptography-3.4.7 docker-5.0.0 fastapi-0.68.0 google-auth-1.34.0 grpcio-1.39.0 h11-0.12.0 idna-3.2 importlib-metadata-4.6.3 janus-0.6.1 kubernetes-17.17.0 mlops-tempo-0.3.0 mlserver-0.3.2 multidict-5.1.0 numpy-1.21.1 oauthlib-3.1.1 orjson-3.6.1 packaging-21.0 protobuf-3.17.3 pyasn1-0.4.8 pyasn1-modules-0.2.8 pycparser-2.20 pydantic-1.8.2 pyparsing-2.4.7 python-dateutil-2.8.2 python-rclone-0.0.2 pyyaml-5.4.1 redis-3.5.3 requests-2.26.0 requests-oauthlib-1.3.0 rsa-4.7.2 seldon-deploy-sdk-1.3.0 six-1.16.0 starlette-0.14.2 typing-extensions-3.10.0.0 urllib3-1.26.6 uvicorn-0.14.0 websocket-client-1.1.1 yarl-1.6.3 zipp-3.5.0[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] #[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] # To activate this environment, use[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] #[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] #     $ conda activate tempo-a1d8e8bb-4b2e-4017-8b60-a5bd0a95a426[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] #[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] # To deactivate an active environment, use[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] #[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] #     $ conda deactivate[0m
    [35m2021-08-07 18:30:16.096 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:16.449 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Collecting packages...[0m
    [35m2021-08-07 18:30:17.033 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Packing environment at '/opt/conda/envs/tempo-a1d8e8bb-4b2e-4017-8b60-a5bd0a95a426' to '/tmp/tmp3qf9kl98/environment.tar.gz'[0m
    [########################################] | 100% Completed | 10.8s[0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] 
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] ## Package Plan ##[0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   environment location: /opt/conda/envs/tempo-a1d8e8bb-4b2e-4017-8b60-a5bd0a95a426[0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] The following packages will be REMOVED:[0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   _libgcc_mutex-0.1-main[0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   ca-certificates-2021.7.5-h06a4308_1[0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   certifi-2021.5.30-py37h06a4308_0[0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   ld_impl_linux-64-2.35.1-h7274673_9[0m
    [35m2021-08-07 18:30:28.297 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   libffi-3.3-he6710b0_2[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   libgcc-ng-9.1.0-hdf63c60_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   libstdcxx-ng-9.1.0-hdf63c60_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   ncurses-6.2-he6710b0_1[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   openssl-1.1.1k-h27cfd23_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   pip-21.2.2-py37h06a4308_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   python-3.7.9-h7579374_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   readline-8.1-h27cfd23_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   setuptools-52.0.0-py37h06a4308_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   sqlite-3.36.0-hc218d9a_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   tk-8.6.10-hbc83047_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   wheel-0.36.2-pyhd3eb1b0_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   xz-5.2.5-h7b6447c_0[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51]   zlib-1.2.11-h7b6447c_3[0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:28.298 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:28.333 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Preparing transaction: ...working... done[0m
    [35m2021-08-07 18:30:28.432 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Verifying transaction: ...working... done[0m
    [35m2021-08-07 18:30:28.616 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Executing transaction: ...working... done[0m
    [35m2021-08-07 18:30:28.193 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:28.194 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Remove all packages in environment /opt/conda/envs/tempo-a1d8e8bb-4b2e-4017-8b60-a5bd0a95a426:[0m
    [35m2021-08-07 18:30:28.194 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51][0m
    [35m2021-08-07 18:30:59.653 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Insights Manager not initialised as empty URL provided.[0m
    [35m2021-08-07 18:31:09.816 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] {'output0': array([[0.00847207, 0.03168793, 0.95984   ]], dtype=float32), 'output1': 'xgboost prediction'}[0m
    [35m2021-08-07 18:31:14.000 [0m[32m[28/tempo/162 (pid 2051921)] [0m[22m[319ba9da-f3dc-437e-a37d-d8b50004dd51] Task finished with exit code 0.[0m
    [35m2021-08-07 19:31:14.909 [0m[32m[28/tempo/162 (pid 2051921)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:31:15.927 [0m[32m[28/end/163 (pid 2052740)] [0m[1mTask is starting.[0m
    [35m2021-08-07 18:31:16.953 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Task is starting (status SUBMITTED)...[0m
    [35m2021-08-07 18:31:21.317 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Task is starting (status RUNNABLE)...[0m
    [35m2021-08-07 18:31:26.745 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Task is starting (status STARTING)...[0m
    [35m2021-08-07 18:31:28.906 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Task is starting (status RUNNING)...[0m
    [35m2021-08-07 18:31:27.744 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Setting up task environment.[0m
    [35m2021-08-07 18:31:33.832 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Downloading code package...[0m
    [35m2021-08-07 18:31:34.343 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Code package downloaded.[0m
    [35m2021-08-07 18:31:34.356 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Task is starting.[0m
    [35m2021-08-07 18:31:34.718 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Bootstrapping environment...[0m
    [35m2021-08-07 18:31:50.960 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Environment bootstrapped.[0m
    [35m2021-08-07 18:31:58.917 [0m[32m[28/end/163 (pid 2052740)] [0m[22m[e251303f-114e-40cd-a108-7814bd64e0e7] Task finished with exit code 0.[0m
    [35m2021-08-07 19:32:01.171 [0m[32m[28/end/163 (pid 2052740)] [0m[1mTask finished successfully.[0m
    [35m2021-08-07 19:32:01.645 [0m[1mDone![0m


Use the saved client from the Flow to make predictions


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




```python

```
