import os
import time
from http import cookies
from typing import Dict, Sequence

from seldon_deploy_sdk import ApiClient, Configuration, EnvironmentApi, SeldonDeploymentsApi
from seldon_deploy_sdk.auth import OIDCAuthenticator, SessionAuthenticator
from seldon_deploy_sdk.models.object_meta import ObjectMeta
from seldon_deploy_sdk.models.predictive_unit import PredictiveUnit
from seldon_deploy_sdk.models.predictor_spec import PredictorSpec
from seldon_deploy_sdk.models.seldon_deployment import SeldonDeployment
from seldon_deploy_sdk.models.seldon_deployment_spec import SeldonDeploymentSpec

from tempo.seldon.endpoint import Endpoint
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.seldon.specs import KubernetesSpec
from tempo.serve.base import ClientModel, ModelSpec, Runtime
from tempo.serve.metadata import EnterpriseRuntimeAuthType, EnterpriseRuntimeOptions


# FIXME: Needs updating for runtime options to include EnterpriseRuntimeOptions
class SeldonDeployRuntime(Runtime):
    def list_models(self) -> Sequence[ClientModel]:
        pass

    def __init__(self):
        self.api_client = None

    def authenticate(self, settings: EnterpriseRuntimeOptions):
        config = Configuration()
        config.host = settings.host
        config.verify_ssl = settings.verify_ssl
        if settings.auth_type == EnterpriseRuntimeAuthType.session_cookie:
            auth = SessionAuthenticator(config)
            # TODO can we use just cookie_str in API client
            cookie_str = auth.authenticate(settings.user, settings.password)
            ckies: cookies.SimpleCookie = cookies.SimpleCookie()
            ckies.load(cookie_str)
            token = ckies["authservice_session"].value
            cookie_str = f"authservice_session={token}"
            api_client = ApiClient(config, cookie=cookie_str)
            self.api_client = api_client
        elif settings.auth_type == EnterpriseRuntimeAuthType.oidc:
            if not settings.verify_ssl:
                os.environ["CURL_CA_BUNDLE"] = ""
            config.oidc_client_id = settings.oidc_client_id
            config.oidc_server = settings.oidc_server
            auth = OIDCAuthenticator(config)
            config.access_token = auth.authenticate(settings.user, settings.password)
            api_client = ApiClient(config)
            self.api_client = api_client
        else:
            raise ValueError(f"Unknown auth type {settings.auth_type}")

    def _create_api_client(self):
        if self.api_client is not None:
            return self.api_client
        else:
            # TODO: auto authenticate from environment
            pass

    def deploy_spec(self, model_spec: ModelSpec):
        self._create_api_client()
        env_api = EnvironmentApi(self.api_client)
        user = env_api.read_user()

        print(user)

        # TODO: Use KubernetesSpec().spec
        sd = SeldonDeployment(
            kind="SeldonDeployment",
            api_version="machinelearning.seldon.io/v1",
            metadata=ObjectMeta(
                name=model_spec.model_details.name,
                namespace=model_spec.runtime_options.namespace,  # type: ignore
            ),
            spec=SeldonDeploymentSpec(
                predictors=[
                    PredictorSpec(
                        graph=PredictiveUnit(
                            implementation=KubernetesSpec.Implementations[model_spec.model_details.platform],
                            model_uri=model_spec.model_details.uri,
                            name="model",
                        ),
                        name="default",
                        replicas=model_spec.runtime_options.replicas,  # type: ignore
                    )
                ]
            ),
        )

        dep_instance = SeldonDeploymentsApi(self.api_client)
        dep_instance.create_seldon_deployment(model_spec.runtime_options.namespace, sd)  # type: ignore

    def wait_ready_spec(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        self._create_api_client()
        ready = False
        t0 = time.time()
        dep_instance = SeldonDeploymentsApi(self.api_client)
        while not ready:
            sdep: SeldonDeployment = dep_instance.read_seldon_deployment(
                model_spec.model_details.name,
                model_spec.runtime_options.namespace,  # type: ignore
            )
            sdep_dict = sdep.to_dict()
            ready = sdep_dict["status"]["state"] == "Available"
            if timeout_secs is not None:
                t1 = time.time()
                if t1 - t0 > timeout_secs:
                    return ready
        return ready

    def undeploy_spec(self, model_spec: ModelSpec):
        self._create_api_client()
        dep_instance = SeldonDeploymentsApi(self.api_client)
        dep_instance.delete_seldon_deployment(
            model_spec.model_details.name,
            model_spec.runtime_options.namespace,  # type: ignore
            _preload_content=False,
        )

    def get_endpoint_spec(self, model_spec: ModelSpec):
        endpoint = Endpoint()
        return endpoint.get_url(model_spec)

    def get_headers(self, model_spec: ModelSpec) -> Dict[str, str]:
        return {}

    def to_k8s_yaml_spec(self, model_spec: ModelSpec) -> str:
        srt = SeldonKubernetesRuntime()
        return srt.to_k8s_yaml_spec(model_spec)
