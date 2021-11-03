import numpy as np
import pytest

from tempo.protocols.seldon import SeldonProtocol
from tempo.seldon.deploy import SeldonDeployRuntime
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.serve.metadata import (
    EnterpriseRuntimeAuthType,
    EnterpriseRuntimeOptions,
    IngressOptions,
    KubernetesRuntimeOptions,
    ModelFramework,
)
from tempo.serve.model import Model


@pytest.mark.skip("needs deploy cluster")
def test_deploy():
    rt = SeldonDeployRuntime()

    config = EnterpriseRuntimeOptions(
        host="https://34.78.44.92/seldon-deploy/api/v1alpha1",
        user="admin@seldon.io",
        password="12341234",
        oidc_server="https://34.78.44.92/auth/realms/deploy-realm",
        oidc_client_id="sd-api",
        verify_ssl=False,
        auth_type=EnterpriseRuntimeAuthType.oidc,
    )

    rt.authenticate(settings=config)

    options = KubernetesRuntimeOptions(
        namespace="seldon",
        ingress_options=IngressOptions(ssl=True, verify_ssl=False),
    )

    sklearn_model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris",
        protocol=SeldonProtocol(),
        runtime_options=options,
    )

    rt.deploy(sklearn_model)
    rt.wait_ready(sklearn_model)
    print(sklearn_model(np.array([[4.9, 3.1, 1.5, 0.2]])))
    rt.undeploy(sklearn_model)


@pytest.mark.skip("needs deploy cluster")
def test_deploy_yaml():
    rt = SeldonDeployRuntime()

    config = EnterpriseRuntimeOptions(
        host="https://34.78.44.92/seldon-deploy/api/v1alpha1",
        user="admin@seldon.io",
        password="12341234",
        oidc_server="https://34.78.44.92/auth/realms/deploy-realm",
        oidc_client_id="sd-api",
        verify_ssl=False,
        auth_type=EnterpriseRuntimeAuthType.oidc,
    )

    rt.authenticate(settings=config)

    options = KubernetesRuntimeOptions(
        namespace="seldon",
        ingress_options=IngressOptions(ssl=True, verify_ssl=False),
    )

    sklearn_model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris",
        protocol=SeldonProtocol(),
        runtime_options=options,
    )

    spec = rt.manifest(sklearn_model)
    rtk = SeldonKubernetesRuntime()
    expected = rtk.manifest(sklearn_model)
    assert spec == expected
