
![GitHub](https://img.shields.io/badge/Version-0.1.0-green.svg)
![GitHub](https://img.shields.io/badge/Python-3.5—3.8-blue.svg)
![GitHub](https://img.shields.io/badge/License-Apache-black.svg)

# ⏳ Tempo: The MLOps Software Development Kit

## Vision

*Enable data scientists to see a productionised machine learning model within moments, not months. Easy to work with locally and also in kubernetes, whatever your preferred data science tools*

## Highlights

Tempo provides a unified interface to multiple MLOps projects that enable data scientists to deploy and productionise machine learning systems.


 * Package your trained model artifacts to optimized server runtimes (Tensorflow, PyTorch, Sklearn, XGBoost etc)
 * Package custom business logic to production servers.
 * Build an inference pipeline of models and orchestration steps.
 * Include any custom python components as needed. Examples:
     * Outlier detectors with Alibi-Detect.
     * Explainers with Alibi-Explain.
 * Test Locally - Deploy to Production
     * Run with local unit tests.     
     * Deploy locally to Docker to test with Docker runtimes.
     * Deploy to production on Kubernetes
     * Extract declarative Kubernetes yaml to follow GitOps workflows.     
 * Supporting a wide range of production runtimes
     * Seldon Core open source
     * KFServing open source
     * Seldon Deploy enterprise     
 * Create stateful services. Examples:
    * Multi-Armed Bandits.



## Workflow

1. Develop locally.
2. Test locally on Docker with production artifacts.
3. Push artifacts to remote bucket store and launch remotely (on Kubernetes).

![overview](https://raw.githubusercontent.com/SeldonIO/tempo/master/docs/assets/tempo-overview.png)

## Motivating Synopsis

Data scientists can easily test their models and orchestrate them with pipelines.

Below we see two `Model`s (sklearn and xgboost) with a function decorated `pipeline` to call both.


```python
sklearn_model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        protocol=SeldonProtocol(),
        local_folder=SKLEARN_FOLDER,
        uri="s3://tempo/basic/sklearn"
)

xgboost_model = Model(
        name="test-iris-xgboost",
        platform=ModelFramework.XGBoost,
        protocol=SeldonProtocol(),
        local_folder=XGBOOST_FOLDER,
        uri="s3://tempo/basic/xgboost"
)

@pipeline(name="classifier",
          uri="s3://tempo/basic/pipeline",
          local_folder=PIPELINE_ARTIFACTS_FOLDER,
          models=[sklearn_model, xgboost_model])
def classifier(payload: np.ndarray) -> Tuple[np.ndarray,str]:
    res1 = sklearn_model(payload)

    if res1[0][0] > 0.5:
        return res1,"sklearn prediction"
    else:
        return xgboost_model(payload),"xgboost prediction"
```

Save the pipeline code.

```
save(classifier, save_env=True)
```

Deploy to docker.

```
docker_runtime = SeldonDockerRuntime()
docker_runtime.deploy(classifier)
docker_runtime.wait_ready(classifier)
```

Make predictions on containerized servers that would be used in production.

```
classifier.remote(payload=np.array([[1, 2, 3, 4]]))
```

Deploy to Kubernetes for production.

```
k8s_runtime = SeldonKubernetesRuntime()
k8s_runtime.deploy(classifier)
k8s_runtime.wait_ready(classifier)
```

This is an extract from the two intridyctory examples for [local](https://tempo.readthedocs.io/en/latest/examples/intro/local.html) and [Kubernetes](https://tempo.readthedocs.io/en/latest/examples/intro/k8s.html) demos.