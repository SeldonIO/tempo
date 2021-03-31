# Serving a Custom Model

This example walks you through how to deploy a custom model with Tempo.
In particular, we will walk you through how to write custom logic to run inference on a [`numpyro` model](http://num.pyro.ai/en/stable/).

Note that we've picked `numpyro` for this example simply because it's not supported out of the box, but it should be possible to adapt this example easily to any other custom model.

## Environment

The first step will be to describe our required environment through a `conda.yaml` file, following Conda's syntax.
This will make sure that our dependencies are defined explicitly, so that they can be **shared between our training and inference environments**.


```python
%%writefile ./artifacts/conda.yaml
name: tempo-numpyro
channels:
  - defaults
dependencies:
  - pip=21.0.1
  - python=3.7.9
    
  - pandas=1.0.1
  - pip:
    - mlops-tempo
    - numpyro==0.6.0
    - mlserver==0.3.1.dev5
    - mlserver-tempo==0.3.1.dev5
```

Note that this environment will need to be created before running this notebook.
This can be done by running:

```bash
$ conda env create --name tempo-numpyro --file ./artifacts/conda.yaml
```

## Training

The first step will be to train our model.
This will be a very simple bayesian regression model, based on an example provided in the [`numpyro` docs](https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/source/bayesian_regression.ipynb).

Since this is a probabilistic model, during training we will compute an approximation to the posterior distribution of our model using MCMC.


```python
# Original source code and more details can be found in:
# https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/source/bayesian_regression.ipynb


import numpyro
import numpy as np
import pandas as pd

from numpyro import distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

DATASET_URL = 'https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv'
dset = pd.read_csv(DATASET_URL, sep=';')

standardize = lambda x: (x - x.mean()) / x.std()

dset['AgeScaled'] = dset.MedianAgeMarriage.pipe(standardize)
dset['MarriageScaled'] = dset.Marriage.pipe(standardize)
dset['DivorceScaled'] = dset.Divorce.pipe(standardize)

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

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

num_warmup, num_samples = 1000, 2000

# Run NUTS.
kernel = NUTS(model_function)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, marriage=dset.MarriageScaled.values, divorce=dset.DivorceScaled.values)
mcmc.print_summary()
```

### Saving trained model

Now that we have _trained_ our model, the next step will be to save it so that it can be loaded afterwards at serving-time.
Note that, since this is a probabilistic model, we will only need to save the traces that approximate the posterior distribution over latent parameters.

This will get saved in a `numpyro-divorce.json` file.


```python
import json

samples = mcmc.get_samples()
serialisable = {}
for k, v in samples.items():
    serialisable[k] = np.asarray(v).tolist()
    
model_file_name = "./artifacts/numpyro-divorce.json"
with open(model_file_name, 'w') as model_file:
    json.dump(serialisable, model_file)
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
import os
import json

from numpyro.infer import Predictive
from numpyro import distributions as dist
from jax import random
from tempo import model, ModelFramework

@model(
    name='numpyro-divorce',
    platform=ModelFramework.Custom,
    local_folder="./artifacts",
)
def numpyro_divorce(marriage: np.ndarray, age: np.ndarray) -> np.ndarray:
    rng_key = random.PRNGKey(0)
    predictions = numpyro_divorce.ctx.predictive_dist(
        rng_key=rng_key,
        marriage=marriage,
        age=age
    )
    
    return predictions['obs'].mean()

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

    numpyro_divorce.ctx.predictive_dist = Predictive(model_function, samples)
```

We can now test our custom logic by running inference locally.


```python
marriage = np.array([28.0])
age = np.array([63])
pred = numpyro_divorce(marriage=marriage, age=age)

print(pred)
```

### Deploying model

Finally, we'll be able to deploy our model using Tempo against one of the available runtimes (i.e. Kubernetes, Docker or Seldon Deploy).
For this example, we will deploy the model using the Docker runtime.


```python
from tempo.seldon import SeldonDockerRuntime
from tempo.kfserving import KFServingV2Protocol

docker_runtime = SeldonDockerRuntime(protocol=KFServingV2Protocol())

numpyro_divorce.set_runtime(docker_runtime)
numpyro_divorce.save()
numpyro_divorce.deploy()
numpyro_divorce.wait_ready()
```


