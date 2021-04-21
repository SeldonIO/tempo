# Tempo Workflow

## Do your data science

Do your data science and create the models for your application!


## Define Tempo Artifacts

The next steps is to:

 1. Associate Tempo Models to each model you have created.
 2. Orchestrate your models (if needed) by a Tempo Pipeline


### Tempo Models

Tempo Models can be defined:

 1. Using the [Model class](../api/tempo.serve.model.html) for standard artifacts you want to run in a prepackaged server provided by one of the Tempo Runtimes.
 2. Using the [model decorator](../api/tempo.serve.utils.html) to decorate a function or class with custom code to load and run your model.


#### Using the Model class

Use this when you have a standard artifact that falls into one of the [ModelFramework](api/tempo.serve.metadata.html#tempo.serve.metadata.ModelFramework) supported by Tempo and this ModelFramework is supported by one of the Runtimes. Example:

```python
    sklearn_model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        local_folder=f"{artifacts_folder}/{SKLearnFolder}",
        uri="s3://tempo/basic/sklearn",
    )
```

For further details see the [Model class](../api/tempo.serve.model.html) docs.


#### Using the model decorator

The model decorator can be used to create a Tempo model from custom python code. Use this if you want to manage the serving of your model yourself as it can not be run on one of the out of the box servers provided by the Runtimes. An example for this is a custom outlier detector written using Seldon's Alibi-Detect library:

```python
    @model(
        name="outlier",
        platform=ModelFramework.TempoPipeline,
        protocol=KFServingV2Protocol(),
        uri="s3://tempo/outlier/cifar10/outlier",
        local_folder=f"{artifacts_folder}/{OUTLIER_FOLDER}",
    )
    class OutlierModel(object):

        def __init__(self):
            self.loaded = False

        def load(self):
            from alibi_detect.utils.saving import load_detector

            if "MLSERVER_MODELS_DIR" in os.environ:
                models_folder = "/mnt/models"
            else:
                models_folder = f"{artifacts_folder}/{OUTLIER_FOLDER}"
            print(f"Loading from {models_folder}")
            self.od = load_detector(f"{models_folder}/cifar10")
            self.loaded = True

        def unload(self):
            self.od = None
            self.loaded = False

        @predictmethod
        def outlier(self, payload: np.ndarray) -> dict:
            if not self.loaded:
                self.load()
            od_preds = self.od.predict(
                payload,
                outlier_type="instance",  # use 'feature' or 'instance' level
                return_feature_score=True,
                # scores used to determine outliers
                return_instance_score=True,
            )

            return json.loads(json.dumps(od_preds, cls=OutlierModel.NumpyEncoder))

```

The above example decorates a class with the predict method defined by the `@predictmethod` function decorator. The class contains code to load a saved outlier detector, test if an input is an outlier and return the result as json.

For further details see the [model class](../api/tempo.serve.utils.html).

### Tempo Pipelines

Pipelines allow you to orchestrate models using any custom python code you need for your business logic. They have a similar structure to the model decorator discussed above. An example is shown below from the [intro example](../examples/intro/README.html):

```python
def get_tempo_artifacts(artifacts_folder: str) -> Tuple[Pipeline, Model, Model]:

    sklearn_model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        local_folder=f"{artifacts_folder}/{SKLearnFolder}",
        uri="s3://tempo/basic/sklearn",
    )

    xgboost_model = Model(
        name="test-iris-xgboost",
        platform=ModelFramework.XGBoost,
        local_folder=f"{artifacts_folder}/{XGBoostFolder}",
        uri="s3://tempo/basic/xgboost",
    )

    @pipeline(
        name="classifier",
        uri="s3://tempo/basic/pipeline",
        local_folder=f"{artifacts_folder}/{PipelineFolder}",
        models=PipelineModels(sklearn=sklearn_model, xgboost=xgboost_model),
    )
    def classifier(payload: np.ndarray) -> Tuple[np.ndarray, str]:
        res1 = classifier.models.sklearn(input=payload)

        if res1[0] == 1:
            return res1, SKLearnTag
        else:
            return classifier.models.xgboost(input=payload), XGBoostTag

    return classifier, sklearn_model, xgboost_model
```

The above code does some simple logic against two models - an sklearn model and an xgboost model. As part of the `pipeline` decorator for a function or class you defined the models you want to orchestrate here via:

```python
        models=PipelineModels(sklearn=sklearn_model, xgboost=xgboost_model),
```

For further details see the [pipeline class](../api/tempo.serve.utils.html).

## Save Model Artifacts

For any custom python code defined using the *model* and *pipeline* decorators you will need to save the python environment needed to run the code and the pickled code itself. This can be done by using the [save](../api/tempo.serve.loader.html) method. See the [intro example](../examples/intro/README.html) for a demonstration.

## Deploy model

Once saved you can deploy your artifacts using a Runtime.

### Deploy to Docker

The [SeldonDockerRuntime](../api/tempo.seldon.docker.html) can be used to deploy your pipeline to Docker as for example below taken from the [intro example](../examples/intro/README.html).

```
from tempo.seldon.docker import SeldonDockerRuntime
docker_runtime = SeldonDockerRuntime()
docker_runtime.deploy(classifier)
docker_runtime.wait_ready(classifier)
```

### Deploy to Kubernetes

To run your pipelines and models remotely on a Kubernetes cluster you will need to upload those artifacts to remote bucket stores accesible from your Kubernetes cluster. For this we provide [upload](../api/tempo.serve.loader.html) methods that utilize rclone to achieve this. An example is shown below from our [intro example](../examples/intro/README.html):

```
from tempo.serve.loader import upload
upload(sklearn_model)
upload(xgboost_model)
upload(classifier)
```

Once uploaded you can run your pipelines you can deploy to Kubernetes in two main ways.

#### Deploy to Kubernetes Directly

For Kubernetes you can use a Kubernetes Runtime such as [SeldonKubernetesRuntime](../api/tempo.seldon.k8s.html) or [KFServingKubernetesRuntime](../api/tempo.kfserving.k8s.html). 

Create appropriate Kubernetes settings as shown below for your use case. This may require creating the appropriate RBAC to allow components to access the remote bucket storage.

```
from tempo.serve.metadata import RuntimeOptions, KubernetesOptions
runtime_options = RuntimeOptions(
        k8s_options=KubernetesOptions(
            namespace="production",
            authSecretName="minio-secret"
        )
    )

```

The you can deploy directly from tempo. In the example below using the [SeldonKubernetesRuntime](../api/tempo.seldon.k8s.html).

```
from tempo.seldon.k8s import SeldonKubernetesRuntime
k8s_runtime = SeldonKubernetesRuntime(runtime_options)
k8s_runtime.deploy(classifier)
k8s_runtime.wait_ready(classifier)
```


#### Deploy from YAML

Alternatively you can use GitOps principles and generate the appropriate yaml which can be stored on source control and updated via your production/devops continuous deployment process. For this Runtimes can implement `to_k8s_yaml` methods which can be later modified via Kustomize or other processes for production settings. For an example see the [intro example](../examples/intro/README.html).
