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

For further details see the [Model class definition](../api/tempo.serve.model.html) docs.


#### Using the model decorator

The model decorator can be used to create a Tempo model from custom python code. Use this if you want to manage the serving of your model yourself as it can not be run on one of the out of the box servers provided by the Runtimes. An example for this is a custom outlier detector written using Seldon's Alibi-Detect library:

```python
def create_outlier_cls():
    @model(
        name="outlier",
        platform=ModelFramework.Custom,
        protocol=KFServingV2Protocol(),
        uri="s3://tempo/outlier/cifar10/outlier",
        local_folder=os.path.join(ARTIFACTS_FOLDER, OUTLIER_FOLDER),
    )
    class OutlierModel(object):
        def __init__(self):
            from alibi_detect.utils.saving import load_detector

            model = self.get_tempo()
            models_folder = model.details.local_folder
            print(f"Loading from {models_folder}")
            self.od = load_detector(os.path.join(models_folder, "cifar10"))

        @predictmethod
        def outlier(self, payload: np.ndarray) -> dict:
            od_preds = self.od.predict(
                payload,
                outlier_type="instance",  # use 'feature' or 'instance' level
                return_feature_score=True,
                # scores used to determine outliers
                return_instance_score=True,
            )

            return json.loads(json.dumps(od_preds, cls=NumpyEncoder))

    return OutlierModel
```

The above example decorates a class with the predict method defined by the `@predictmethod` function decorator. The class contains code to load a saved outlier detector, test if an input is an outlier and return the result as json.

For further details see the [Model definition](../api/tempo.serve.utils.html).


An alternative is to decorate a function. This is shown below from our [custom model example](../examples/custom-model/README.html):

```python
def get_tempo_artifact(local_folder: str):
    @model(
        name="numpyro-divorce",
        platform=ModelFramework.Custom,
        local_folder=local_folder,
        uri="s3://tempo/divorce",
    )
    def numpyro_divorce(marriage: np.ndarray, age: np.ndarray) -> np.ndarray:
        rng_key = random.PRNGKey(0)
        predictions = numpyro_divorce.context.predictive_dist(rng_key=rng_key, marriage=marriage, age=age)

        mean = predictions["obs"].mean(axis=0)
        return np.asarray(mean)

    @numpyro_divorce.loadmethod
    def load_numpyro_divorce():
        model_uri = os.path.join(numpyro_divorce.details.local_folder, "numpyro-divorce.json")

        with open(model_uri) as model_file:
            raw_samples = json.load(model_file)

        samples = {}
        for k, v in raw_samples.items():
            samples[k] = np.array(v)

        numpyro_divorce.context.predictive_dist = Predictive(model_function, samples)

    return numpyro_divorce
```

In the above function we use an auxillary function to allow us to load the model. For this we use a decorator starting with the function name of the form `<function_name>.loadmethod`. Inside this method one can set context variables which can later be accessed from the main function.


### Tempo Pipelines

Pipelines allow you to orchestrate models using any custom python code you need for your business logic. They have a similar structure to the model decorator discussed above. An example is shown below from the [multi-model example](../examples/multi-model/README.html):

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

For any custom python code defined using the *model* and *pipeline* decorators you will need to save the python environment needed to run the code and the pickled code itself. This can be done by using the [save](../api/tempo.serve.loader.html) method. See the [example](../examples/multi-model/README.html) for a demonstration.

## Deploy model

Once saved you can deploy your artifacts using a Runtime.

### Deploy to Docker

By default tempo will deploy to Docker:

```python
from tempo import deploy_local
remote_model = deploy_local(classifier)
```

The returned RemoteModel can be used to get predictions:

```python
remote_model.predict(np.array([[1, 2, 3, 4]]))
```

And then undeploy:

```python
remote_model.undeploy()
```

### Deploy to Production

To run your pipelines and models remotely in production you will need to upload those artifacts to remote bucket stores accesible from your Kubernetes cluster. For this we provide [upload](../api/tempo.serve.loader.html) methods that utilize rclone to achieve this. An example is shown below from our [multi-model example](../examples/multi-model/README.html):

```
from tempo.serve.loader import upload
upload(sklearn_model)
upload(xgboost_model)
upload(classifier)
```

Once uploaded you can run your pipelines you can deploy to production in two main ways.

#### Update RuntimeOptions with the production runtime

For Kubernetes you can use a Kubernetes Runtime such as [SeldonKubernetesRuntime](../api/tempo.seldon.k8s.html).

Create appropriate Kubernetes settings as shown below for your use case. This may require creating the appropriate RBAC to allow components to access the remote bucket storage.

```
from tempo.serve.metadata import SeldonCoreOptions

runtime_options = SeldonCoreOptions(**{
    "remote_options": {
        "namespace": "production",
        "authSecretName": "minio-secret"
    }
})	
```

Then you can deploy directly from tempo:

```
from tempo import deploy_remote
remote_model = deploy_remote(classifier, options=runtime_options)
```

And then call prediction as before:

```python
remote_model.predict(np.array([[1, 2, 3, 4]]))
```

You can also undeploy:

```python
remote_model.undeploy()
```

#### GitOps

Alternatively you can use GitOps principles and generate the appropriate yaml which can be stored on source control and updated via your production/devops continuous deployment process. For this Runtimes can implement the `manifest` method which can be later modified via Kustomize or other processes for production settings. For an example see the [multi-model example](../examples/multi-model/README.html).
