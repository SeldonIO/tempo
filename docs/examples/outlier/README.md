# Outlier Example

![architecture](architecture.png)

In this example we will:

  * [Describe the project structure](#Project-Structure)
  * [Train some models](#Train-Models)
  * [Create Tempo artifacts](#Create-Tempo-Artifacts)
  * [Run unit tests](#Unit-Tests)
  * [Save python environment for our classifier](#Save-Classifier-Environment)
  * [Test Locally on Docker](#Test-Locally-on-Docker)
  * [Production on Kubernetes via Tempo](#Production-Option-1-(Deploy-to-Kubernetes-with-Tempo))
  * [Prodiuction on Kuebrnetes via GitOps](#Production-Option-2-(Gitops))

## Prerequisites

This notebooks needs to be run in the `tempo-examples` conda environment defined below. Create from project root folder:

```bash
conda env create --name tempo-examples --file conda/tempo-examples.yaml
```

## Project Structure


```python
!tree -P "*.py"  -I "__init__.py|__pycache__" -L 2
```

    [01;34m.[00m
    ├── [01;34martifacts[00m
    │   ├── [01;34mmodel[00m
    │   ├── [01;34moutlier[00m
    │   └── [01;34msvc[00m
    ├── [01;34mk8s[00m
    │   └── [01;34mrbac[00m
    ├── [01;34msrc[00m
    │   ├── constants.py
    │   ├── data.py
    │   ├── outlier.py
    │   ├── tempo.py
    │   └── utils.py
    └── [01;34mtests[00m
        └── test_tempo.py
    
    8 directories, 6 files


## Train Models

 * This section is where as a data scientist you do your work of training models and creating artfacts.
 * For this example we train sklearn and xgboost classification models for the iris dataset.


```python
import os
import logging
import numpy as np
import tempo

from tempo.utils import logger
from src.constants import ARTIFACTS_FOLDER

logger.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
```


```python
from src.data import Cifar10
data = Cifar10()
```

    (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)


Download pretrained Resnet32 Tensorflow model for CIFAR10


```python
!rclone --config ./rclone-gcs.conf copy gs://seldon-models/tfserving/cifar10/resnet32 ./artifacts/model
```

Download or train an outlier detector on CIFAR10 data


```python
load_pretrained = True
if load_pretrained:  # load pre-trained detector
    !rclone --config ./rclone-gcs.conf copy gs://seldon-models/tempo/cifar10/outlier/cifar10 ./artifacts/outlier/cifar10
else:
    from src.outlier import train_outlier_detector
    train_outlier_detector(data, ARTIFACTS_FOLDER)
```

## Create Tempo Artifacts



```python
from src.tempo import create_outlier_cls, create_model, create_svc_cls

cifar10_model = create_model()
OutlierModel = create_outlier_cls()
outlier = OutlierModel()
Cifar10Svc = create_svc_cls(outlier, cifar10_model)
svc = Cifar10Svc()
```

    ERROR:fbprophet:Importing plotly failed. Interactive plots will not work.


    Loading from /home/alejandro/Programming/kubernetes/seldon/tempo/docs/examples/outlier/artifacts/outlier



```python
# %load src/tempo.py
import json
import os

import numpy as np
from alibi_detect.base import NumpyEncoder
from src.constants import ARTIFACTS_FOLDER, MODEL_FOLDER, OUTLIER_FOLDER

from tempo.kfserving.protocol import KFServingV1Protocol, KFServingV2Protocol
from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model
from tempo.serve.pipeline import PipelineModels
from tempo.serve.utils import model, pipeline, predictmethod


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


def create_model():

    cifar10_model = Model(
        name="resnet32",
        protocol=KFServingV1Protocol(),
        platform=ModelFramework.Tensorflow,
        uri="gs://seldon-models/tfserving/cifar10/resnet32",
        local_folder=os.path.join(ARTIFACTS_FOLDER, MODEL_FOLDER),
    )

    return cifar10_model


def create_svc_cls(outlier, model):
    @pipeline(
        name="cifar10-service",
        protocol=KFServingV2Protocol(),
        uri="s3://tempo/outlier/cifar10/svc",
        local_folder=os.path.join(ARTIFACTS_FOLDER, "svc"),
        models=PipelineModels(outlier=outlier, cifar10=model),
    )
    class Cifar10Svc(object):
        @predictmethod
        def predict(self, payload: np.ndarray) -> np.ndarray:
            r = self.models.outlier(payload=payload)
            if r["data"]["is_outlier"][0]:
                return np.array([])
            else:
                return self.models.cifar10(payload)

    return Cifar10Svc

```

## Unit Tests

 * Here we run our unit tests to ensure the orchestration works before running on the actual models.


```python
# %load tests/test_tempo.py
import numpy as np
from src.tempo import create_model, create_outlier_cls, create_svc_cls


def test_svc_outlier():
    model = create_model()
    OutlierModel = create_outlier_cls()
    outlier = OutlierModel()
    Cifar10Svc = create_svc_cls(outlier, model)
    svc = Cifar10Svc()
    svc.models.outlier = lambda payload: {"data": {"is_outlier": [1]}}
    svc.models.cifar10 = lambda input: np.array([[0.2]])
    res = svc(np.array([1]))
    assert res.shape[0] == 0


def test_svc_inlier():
    model = create_model()
    OutlierModel = create_outlier_cls()
    outlier = OutlierModel()
    Cifar10Svc = create_svc_cls(outlier, model)
    svc = Cifar10Svc()
    svc.models.outlier = lambda payload: {"data": {"is_outlier": [0]}}
    svc.models.cifar10 = lambda input: np.array([[0.2]])
    res = svc(np.array([1]))
    assert res.shape[0] == 1

```


```python
!python -m pytest tests/
```

    [1mTest session starts (platform: linux, Python 3.7.10, pytest 5.3.1, pytest-sugar 0.9.4)[0m
    rootdir: /home/alejandro/Programming/kubernetes/seldon/tempo, inifile: setup.cfg
    plugins: cases-3.4.6, sugar-0.9.4, xdist-1.30.0, anyio-3.2.1, requests-mock-1.7.0, django-3.8.0, forked-1.1.3, flaky-3.6.1, asyncio-0.14.0, celery-4.4.0, cov-2.8.1
    [1mcollecting ... [0m
     [36mdocs/examples/outlier/tests/[0mtest_tempo.py[0m [32m✓[0m[32m✓[0m                    [32m100% [0m[40m[32m█[0m[40m[32m████[0m[40m[32m█[0m[40m[32m████[0m
    [33m=============================== warnings summary ===============================[0m
    /home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/autograph/impl/api.py:22
      /home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/autograph/impl/api.py:22: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
        import imp
    
    /home/alejandro/miniconda3/lib/python3.7/site-packages/packaging/version.py:130
      /home/alejandro/miniconda3/lib/python3.7/site-packages/packaging/version.py:130: DeprecationWarning: Creating a LegacyVersion has been deprecated and will be removed in the next major release
        DeprecationWarning,
    
    -- Docs: https://docs.pytest.org/en/latest/warnings.html
    
    Results (5.21s):
    [32m       2 passed[0m
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/matplotlib/_pylab_helpers.py", line 77, in destroy_all
        gc.collect(1)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_mean.kernel'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/matplotlib/_pylab_helpers.py", line 77, in destroy_all
        gc.collect(1)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_mean.kernel'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/matplotlib/_pylab_helpers.py", line 77, in destroy_all
        gc.collect(1)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_mean.bias'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/matplotlib/_pylab_helpers.py", line 77, in destroy_all
        gc.collect(1)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_mean.bias'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/matplotlib/_pylab_helpers.py", line 77, in destroy_all
        gc.collect(1)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_log_var.kernel'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/matplotlib/_pylab_helpers.py", line 77, in destroy_all
        gc.collect(1)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_log_var.kernel'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/matplotlib/_pylab_helpers.py", line 77, in destroy_all
        gc.collect(1)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_log_var.bias'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/matplotlib/_pylab_helpers.py", line 77, in destroy_all
        gc.collect(1)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_log_var.bias'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/matplotlib/_pylab_helpers.py", line 77, in destroy_all
        gc.collect(1)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 169, in __del__
        "A checkpoint was restored (e.g. tf.train.Checkpoint.restore or "
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/matplotlib/_pylab_helpers.py", line 77, in destroy_all
        gc.collect(1)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 169, in __del__
        "A checkpoint was restored (e.g. tf.train.Checkpoint.restore or "
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_mean.kernel'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_mean.kernel'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_mean.bias'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_mean.bias'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_log_var.kernel'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_log_var.kernel'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_log_var.bias'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 161, in __del__
        .format(pretty_printer.node_names[node_id]))
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'Unresolved object in checkpoint: (root).encoder.fc_log_var.bias'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 169, in __del__
        "A checkpoint was restored (e.g. tf.train.Checkpoint.restore or "
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.'
    Arguments: ()
    --- Logging error ---
    Traceback (most recent call last):
      File "/home/alejandro/miniconda3/lib/python3.7/logging/__init__.py", line 1028, in emit
        stream.write(msg + self.terminator)
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/_pytest/capture.py", line 427, in write
        self.buffer.write(obj)
    ValueError: I/O operation on closed file
    Call stack:
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/tracking/util.py", line 169, in __del__
        "A checkpoint was restored (e.g. tf.train.Checkpoint.restore or "
      File "/home/alejandro/miniconda3/lib/python3.7/site-packages/tensorflow/python/platform/tf_logging.py", line 178, in warning
        get_logger().warning(msg, *args, **kwargs)
    Message: 'A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.'
    Arguments: ()


## Save Outlier and Svc Environments



```python
%%writefile artifacts/outlier/conda.yaml
name: tempo
channels:
  - defaults
dependencies:
  - python=3.7.9
  - pip:
    - alibi-detect==0.6.2
    - dill==0.3.2
    - opencv-python-headless
    - mlops-tempo @ file:///home/alejandro/Programming/kubernetes/seldon/tempo
    - mlserver==0.3.2
```

    Overwriting artifacts/outlier/conda.yaml



```python
%%writefile artifacts/svc/conda.yaml
name: tempo
channels:
  - defaults
dependencies:
  - python=3.7.9
  - pip:
    - mlops-tempo @ file:///home/alejandro/Programming/kubernetes/seldon/tempo
    - mlserver==0.3.2
```

    Overwriting artifacts/svc/conda.yaml



```python
tempo.save(OutlierModel)
```

    Collecting packages...
    Packing environment at '/home/alejandro/miniconda3/envs/tempo-1142b4d9-c66f-47db-bd1a-1eccad0afc0b' to '/home/alejandro/Programming/kubernetes/seldon/tempo/docs/examples/outlier/artifacts/outlier/environment.tar.gz'
    [########################################] | 100% Completed | 56.8s



```python
tempo.save(Cifar10Svc)
```

    Collecting packages...
    Packing environment at '/home/alejandro/miniconda3/envs/tempo-e7f7a0d2-eefe-45cc-8dc9-2be0f74d83a1' to '/home/alejandro/Programming/kubernetes/seldon/tempo/docs/examples/outlier/artifacts/svc/environment.tar.gz'
    [########################################] | 100% Completed | 10.6s


## Test Locally on Docker

Here we test our models using production images but running locally on Docker. This allows us to ensure the final production deployed model will behave as expected when deployed.


```python
from tempo import deploy_local
remote_model = deploy_local(svc)
```


```python
from src.utils import show_image
show_image(data.X_test[0:1])
remote_model.predict(payload=data.X_test[0:1])
```


    
![png](README_files/README_24_0.png)
    


    b'{"model_name":"cifar10-service","model_version":"NOTIMPLEMENTED","id":"7bfbc51d-c042-4517-9d62-4a757eeb0a5f","parameters":null,"outputs":[{"name":"output0","shape":[1,10],"datatype":"FP64","parameters":null,"data":[3.92254496e-09,1.2045546e-11,2.66010169e-09,0.999992609,2.52212834e-10,5.40860242e-07,6.75951833e-06,4.75118165e-12,6.90873403e-09,1.07275378e-11]}]}'





    array([[3.92254496e-09, 1.20455460e-11, 2.66010169e-09, 9.99992609e-01,
            2.52212834e-10, 5.40860242e-07, 6.75951833e-06, 4.75118165e-12,
            6.90873403e-09, 1.07275378e-11]])




```python
from src.utils import create_cifar10_outlier

outlier_img = create_cifar10_outlier(data)
show_image(outlier_img)
remote_model.predict(payload=outlier_img)
```


    
![png](README_files/README_25_0.png)
    


    b'{"model_name":"cifar10-service","model_version":"NOTIMPLEMENTED","id":"6e0124b8-bbb2-4a82-b167-6008ad17c21a","parameters":null,"outputs":[{"name":"output0","shape":[0],"datatype":"FP64","parameters":null,"data":[]}]}'





    array([], dtype=float64)




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

    secret/minio-secret configured
    serviceaccount/tempo-pipeline unchanged
    role.rbac.authorization.k8s.io/tempo-pipeline unchanged
    rolebinding.rbac.authorization.k8s.io/tempo-pipeline-rolebinding unchanged



```python
from tempo.examples.minio import create_minio_rclone
import os

create_minio_rclone(os.getcwd()+"/rclone-minio.conf")
```


```python
tempo.upload(cifar10_model)
tempo.upload(outlier)
tempo.upload(svc)
```


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
remote_model = deploy_remote(svc, options=runtime_options)
```


```python
from src.utils import show_image

show_image(data.X_test[0:1])
remote_model.predict(payload=data.X_test[0:1])
```


    
![png](README_files/README_33_0.png)
    


    b'{"model_name":"cifar10-service","model_version":"NOTIMPLEMENTED","id":"0a638833-4691-4368-9934-0a8b2db0d69c","parameters":null,"outputs":[{"name":"output0","shape":[1,10],"datatype":"FP64","parameters":null,"data":[3.92254496e-09,1.2045546e-11,2.66010169e-09,0.999992609,2.52212834e-10,5.40860242e-07,6.75951833e-06,4.75118165e-12,6.90873403e-09,1.07275378e-11]}]}'





    array([[3.92254496e-09, 1.20455460e-11, 2.66010169e-09, 9.99992609e-01,
            2.52212834e-10, 5.40860242e-07, 6.75951833e-06, 4.75118165e-12,
            6.90873403e-09, 1.07275378e-11]])




```python
from src.utils import create_cifar10_outlier

outlier_img = create_cifar10_outlier(data)
show_image(outlier_img)
remote_model.predict(payload=outlier_img)
```


    
![png](README_files/README_34_0.png)
    


    b'{"model_name":"cifar10-service","model_version":"NOTIMPLEMENTED","id":"4b7a7df0-6f37-4052-b5be-c2cf277bb1ee","parameters":null,"outputs":[{"name":"output0","shape":[0],"datatype":"FP64","parameters":null,"data":[]}]}'





    array([], dtype=float64)




```python
remote_model.undeploy()
```

## Production Option 2 (Gitops)

 * We create yaml to provide to our DevOps team to deploy to a production cluster
 * We add Kustomize patches to modify the base Kubernetes yaml created by Tempo


```python
from tempo.seldon import SeldonKubernetesRuntime

k8s_runtime = SeldonKubernetesRuntime(runtime_options)
yaml_str = k8s_runtime.manifest(svc)

with open(os.getcwd()+"/k8s/tempo.yaml","w") as f:
    f.write(yaml_str)
```


```python
!kustomize build k8s
```


```python

```
