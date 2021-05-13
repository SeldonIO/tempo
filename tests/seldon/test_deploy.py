import os

import numpy as np
import pytest

from tempo.seldon.deploy import SeldonDeployAuthType, SeldonDeployConfig, SeldonDeployRuntime
from tempo.seldon.protocol import SeldonProtocol
from tempo.serve.metadata import IngressOptions, KubernetesOptions, ModelFramework, RuntimeOptions
from tempo.serve.model import Model

TESTS_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(TESTS_PATH, "data")


@pytest.mark.skip("needs deploy cluster")
def test_deploy():
    rt = SeldonDeployRuntime()

    config = SeldonDeployConfig(
        host="https://34.105.240.29",
        user="admin@seldon.io",
        password="12341234",
        oidc_server="https://34.105.240.29/auth/realms/deploy-realm",
        oidc_client_id="sd-api",
        verify_ssl=False,
        auth_type=SeldonDeployAuthType.oidc,
    )

    rt.authenticate(settings=config)

    options = RuntimeOptions(
        runtime="tempo.seldon.SeldonKubernetesRuntime",
        k8s_options=KubernetesOptions(namespace="seldon"),
        ingress_options=IngressOptions(ssl=True, verify_ssl=False),
    )

    sklearn_model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris",
        local_folder=DATA_PATH,
        protocol=SeldonProtocol(),
        runtime_options=options,
        description="A sklearn model",
    )

    rt.deploy(sklearn_model)
    rt.wait_ready(sklearn_model)
    print(sklearn_model(np.array([[4.9, 3.1, 1.5, 0.2]])))
    rt.undeploy(sklearn_model)


@pytest.mark.skip("needs deploy cluster")
def test_deploy_metadata():
    rt = SeldonDeployRuntime()

    config = SeldonDeployConfig(
        host="https://34.105.240.29",
        user="admin@seldon.io",
        password="12341234",
        oidc_server="https://34.105.240.29/auth/realms/deploy-realm",
        oidc_client_id="sd-api",
        verify_ssl=False,
        auth_type=SeldonDeployAuthType.oidc,
    )

    rt.authenticate(settings=config)

    options = RuntimeOptions(
        runtime="tempo.seldon.SeldonKubernetesRuntime",
        k8s_options=KubernetesOptions(namespace="seldon"),
        ingress_options=IngressOptions(ssl=True, verify_ssl=False),
    )

    sklearn_model = Model(
        name="test-iris-sklearn10",
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris10",
        local_folder=DATA_PATH,
        protocol=SeldonProtocol(),
        runtime_options=options,
        description="A sklearn model",
        version="v1.0.0",
        task_type="classification",
    )

    rt.register(sklearn_model)
