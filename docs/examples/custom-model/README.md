# Serving a Custom Model

This example walks you through how to deploy a custom model with Tempo.
In particular, we will walk you through how to write custom logic to run inference on a [numpyro model](http://num.pyro.ai/en/stable/).

Note that we've picked `numpyro` for this example simply because it's not supported out of the box, but it should be possible to adapt this example easily to any other custom model.

## Prerequisites

This notebooks needs to be run in the `tempo-examples` conda environment defined below. Create from project root folder:

```bash
conda env create --name tempo-examples --file conda/tempo-examples.yaml
```

## Project Structure


```python
!tree -P "*.py"  -I "__init__.py|__pycache__" -L 2
```

## Training

The first step will be to train our model.
This will be a very simple bayesian regression model, based on an example provided in the [`numpyro` docs](https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/source/bayesian_regression.ipynb).

Since this is a probabilistic model, during training we will compute an approximation to the posterior distribution of our model using MCMC.


```python
# %load  src/train.py
# Original source code and more details can be found in:
# https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/source/bayesian_regression.ipynb


import numpy as np
import pandas as pd
from jax import random
from numpyro.infer import MCMC, NUTS
from src.tempo import model_function


def train():
    DATASET_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"
    dset = pd.read_csv(DATASET_URL, sep=";")

    def standardize(x):
        (x - x.mean()) / x.std()

    dset["AgeScaled"] = dset.MedianAgeMarriage.pipe(standardize)
    dset["MarriageScaled"] = dset.Marriage.pipe(standardize)
    dset["DivorceScaled"] = dset.Divorce.pipe(standardize)

    # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    num_warmup, num_samples = 1000, 2000

    # Run NUTS.
    kernel = NUTS(model_function)
    mcmc = MCMC(kernel, num_warmup, num_samples)
    mcmc.run(rng_key_, marriage=dset.MarriageScaled.values, divorce=dset.DivorceScaled.values)
    mcmc.print_summary()
    return mcmc


def save(mcmc, folder: str):
    import json

    samples = mcmc.get_samples()
    serialisable = {}
    for k, v in samples.items():
        serialisable[k] = np.asarray(v).tolist()

    model_file_name = f"{folder}/numpyro-divorce.json"
    with open(model_file_name, "w") as model_file:
        json.dump(serialisable, model_file)

```


```python
import os
from tempo.utils import logger
import logging
import numpy as np
logger.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
ARTIFACTS_FOLDER = os.getcwd()+"/artifacts"
from src.train import train, save, model_function
mcmc = train()
```

### Saving trained model

Now that we have _trained_ our model, the next step will be to save it so that it can be loaded afterwards at serving-time.
Note that, since this is a probabilistic model, we will only need to save the traces that approximate the posterior distribution over latent parameters.

This will get saved in a `numpyro-divorce.json` file.


```python
save(mcmc, ARTIFACTS_FOLDER)
```

## Serving

The next step will be to serve our model through Tempo. 
For that, we will implement a custom model to perform inference using our custom `numpyro` model.
Once our custom model is defined, we will be able to deploy it on any of the available runtimes using the same environment that we used for training.

### Custom inference logic 

Our custom model will be responsible of:

- Loading the model from the set samples we saved previously.
- Running inference using our model structure, and the posterior approximated from the samples.

With Tempo, this can be achieved as:


```python
# %load src/tempo.py
import os
import json
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro.infer import Predictive
from jax import random
from tempo import model, ModelFramework

def model_function(marriage : np.ndarray = None, age : np.ndarray = None, divorce : np.ndarray = None):
    a = numpyro.sample('a', dist.Normal(0., 0.2))
    M, A = 0., 0.
    if marriage is not None:
        bM = numpyro.sample('bM', dist.Normal(0., 0.5))
        M = bM * marriage
    if age is not None:
        bA = numpyro.sample('bA', dist.Normal(0., 0.5))
        A = bA * age
    sigma = numpyro.sample('sigma', dist.Exponential(1.))
    mu = a + M + A
    numpyro.sample('obs', dist.Normal(mu, sigma), obs=divorce)


def get_tempo_artifact(local_folder: str):
    @model(
        name='numpyro-divorce',
        platform=ModelFramework.Custom,
        local_folder=local_folder,
        uri="s3://tempo/divorce",
    )
    def numpyro_divorce(marriage: np.ndarray, age: np.ndarray) -> np.ndarray:
        rng_key = random.PRNGKey(0)
        predictions = numpyro_divorce.context.predictive_dist(
            rng_key=rng_key,
            marriage=marriage,
            age=age
        )

        mean = predictions['obs'].mean(axis=0)
        return np.asarray(mean)

    @numpyro_divorce.loadmethod
    def load_numpyro_divorce():
        model_uri = os.path.join(
            numpyro_divorce.details.local_folder,
            "numpyro-divorce.json"
        )

        with open(model_uri) as model_file:
            raw_samples = json.load(model_file)

        samples = {}
        for k, v in raw_samples.items():
            samples[k] = np.array(v)

        print(model_function.__module__)
        numpyro_divorce.context.predictive_dist = Predictive(model_function, samples)

    return numpyro_divorce



```


```python
from src.tempo import get_tempo_artifact
numpyro_divorce = get_tempo_artifact(ARTIFACTS_FOLDER)
```

We can now test our custom logic by running inference locally.


```python
marriage = np.array([28.0])
age = np.array([63])
pred = numpyro_divorce(marriage=marriage, age=age)

print(pred)
```

### Deploy the  Model to Docker

Finally, we'll be able to deploy our model using Tempo against one of the available runtimes (i.e. Kubernetes, Docker or Seldon Deploy).

We'll deploy first to Docker to test.


```python
!cat artifacts/conda.yaml
```


```python
from tempo.serve.loader import save
save(numpyro_divorce)
```


```python
from tempo import deploy
remote_model = deploy(numpyro_divorce)
```

We can now test our model deployed in Docker as:


```python
remote_model.predict(marriage=marriage, age=age)
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


```python
from tempo.examples.minio import create_minio_rclone
import os
create_minio_rclone(os.getcwd()+"/rclone.conf")
```


```python
from tempo.serve.loader import upload
upload(numpyro_divorce)
```


```python
from tempo.serve.metadata import KubernetesOptions
from tempo.seldon.k8s import SeldonCoreOptions
runtime_options = SeldonCoreOptions(
        k8s_options=KubernetesOptions(
            namespace="production",
            authSecretName="minio-secret"
        )
    )
```


```python
from tempo import deploy
remote_model = deploy(numpyro_divorce, options=runtime_options)
```


```python
remote_model.predict(marriage=marriage, age=age)
```


```python
remote_model.undeploy()
```

## Production Option 2 (Gitops)

 * We create yaml to provide to our DevOps team to deploy to a production cluster
 * We add Kustomize patches to modify the base Kubernetes yaml created by Tempo


```python
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.serve.metadata import RuntimeOptions, KubernetesOptions
runtime_options = RuntimeOptions(
        k8s_options=KubernetesOptions(
            namespace="production",
            authSecretName="minio-secret"
        )
    )
k8s_runtime = SeldonKubernetesRuntime(runtime_options)
yaml_str = k8s_runtime.manifest(numpyro_divorce)
with open(os.getcwd()+"/k8s/tempo.yaml","w") as f:
    f.write(yaml_str)
```


```python
!kustomize build k8s
```


```python

```
