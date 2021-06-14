import pytest
import yaml

from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.seldon.protocol import SeldonProtocol
from tempo.serve.metadata import KubernetesOptions, ModelFramework, RuntimeOptions
from tempo.serve.model import Model


@pytest.mark.parametrize(
    "expected",
    [
        """apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  annotations:
    seldon.io/tempo-description: ''
  labels:
    seldon.io/tempo: 'true'
  name: test-iris-sklearn
  namespace: default
spec:
  predictors:
  - graph:
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/sklearn/iris
      name: test-iris-sklearn
      type: MODEL
    name: default
    replicas: 1
  protocol: seldon
"""
    ],
)
def test_seldon_sklearn_model_yaml(expected):
    m = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris",
        local_folder="/tmp/model",
        protocol=SeldonProtocol(),
    )
    runtime = SeldonKubernetesRuntime()
    yaml_str = runtime.manifest(m)
    yaml_obj = yaml.safe_load(yaml_str)
    yaml_obj_expected = yaml.safe_load(expected)
    del yaml_obj["metadata"]["annotations"]["seldon.io/tempo-model"]
    assert yaml_obj == yaml_obj_expected


@pytest.mark.parametrize(
    "expected",
    [
        """apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  annotations:
    seldon.io/tempo-description: ''
  labels:
    seldon.io/tempo: 'true'
  name: test-iris-xgboost
  namespace: default
spec:
  predictors:
  - graph:
      implementation: XGBOOST_SERVER
      modelUri: gs://seldon-models/xgboost/iris
      name: test-iris-xgboost
      type: MODEL
    name: default
    replicas: 1
  protocol: seldon
"""
    ],
)
def test_seldon_xgboost_model_yaml(expected):
    m = Model(
        name="test-iris-xgboost",
        protocol=SeldonProtocol(),
        platform=ModelFramework.XGBoost,
        uri="gs://seldon-models/xgboost/iris",
        local_folder="/tmp/model",
    )
    runtime = SeldonKubernetesRuntime()
    yaml_str = runtime.manifest(m)
    yaml_obj = yaml.safe_load(yaml_str)
    yaml_obj_expected = yaml.safe_load(expected)
    del yaml_obj["metadata"]["annotations"]["seldon.io/tempo-model"]
    assert yaml_obj == yaml_obj_expected


@pytest.mark.parametrize(
    "expected",
    [
        """apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  annotations:
    seldon.io/tempo-description: ''
  labels:
    seldon.io/tempo: 'true'
  name: test-iris-xgboost
  namespace: default
spec:
  predictors:
  - graph:
      envSecretRefName: auth
      implementation: XGBOOST_SERVER
      modelUri: gs://seldon-models/xgboost/iris
      name: test-iris-xgboost
      type: MODEL
    name: default
    replicas: 1
  protocol: seldon
"""
    ],
)
def test_seldon_model_yaml_auth(expected):
    m = Model(
        name="test-iris-xgboost",
        protocol=SeldonProtocol(),
        platform=ModelFramework.XGBoost,
        uri="gs://seldon-models/xgboost/iris",
        local_folder="/tmp/model",
    )
    runtime = SeldonKubernetesRuntime(
        runtime_options=RuntimeOptions(k8s_options=KubernetesOptions(authSecretName="auth"))
    )
    yaml_str = runtime.manifest(m)
    yaml_obj = yaml.safe_load(yaml_str)
    yaml_obj_expected = yaml.safe_load(expected)
    del yaml_obj["metadata"]["annotations"]["seldon.io/tempo-model"]
    assert yaml_obj == yaml_obj_expected
