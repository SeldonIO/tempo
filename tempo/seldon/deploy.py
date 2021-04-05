import os
import time
from enum import Enum
from http import cookies
from typing import Any, Dict

from seldon_deploy_sdk import ApiClient, Configuration, EnvironmentApi, PredictApi, SeldonDeploymentsApi
from seldon_deploy_sdk.auth import OIDCAuthenticator, SessionAuthenticator
from seldon_deploy_sdk.models.object_meta import ObjectMeta
from seldon_deploy_sdk.models.predictive_unit import PredictiveUnit
from seldon_deploy_sdk.models.predictor_spec import PredictorSpec
from seldon_deploy_sdk.models.seldon_deployment import SeldonDeployment
from seldon_deploy_sdk.models.seldon_deployment_spec import SeldonDeploymentSpec

from tempo.seldon.endpoint import Endpoint
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.seldon.specs import KubernetesSpec
from tempo.serve.metadata import ModelDetails
from tempo.serve.remote import Remote
from tempo.serve.runtime import ModelSpec, Runtime


class SeldonDeployAuthType(Enum):
    session_cookie = "session_cookie"
    oidc = "oidc"


class SeldonDeployRuntime(Runtime, Remote):
    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        oidc_server: str = None,
        oidc_client_id: str = None,
        verify_ssl: bool = True,
        auth_type: SeldonDeployAuthType = SeldonDeployAuthType.session_cookie,
    ):
        self._host = host
        self._user = user
        self._password = password
        self._auth_type = auth_type
        self.oidc_client_id = oidc_client_id
        self.oidc_server = oidc_server
        self.verify_ssl = verify_ssl

    def _get_api_client(self):
        config = Configuration()
        config.host = self._host
        config.verify_ssl = self.verify_ssl
        if self._auth_type == SeldonDeployAuthType.session_cookie:
            auth = SessionAuthenticator(config)
            # TODO can we use just cookie_str in API client
            cookie_str = auth.authenticate(self._user, self._password)
            C = cookies.SimpleCookie()
            C.load(cookie_str)
            token = C["authservice_session"].value
            cookie_str = f"authservice_session={token}"
            api_client = ApiClient(config, cookie=cookie_str)
            return api_client
        elif self._auth_type == SeldonDeployAuthType.oidc:
            if not self.verify_ssl:
                os.environ["CURL_CA_BUNDLE"] = ""
            config.oidc_client_id = self.oidc_client_id
            config.oidc_server = self.oidc_server
            auth = OIDCAuthenticator(config)
            config.access_token = auth.authenticate(self._user, self._password)
            api_client = ApiClient(config)
            return api_client
        else:
            raise ValueError(f"Unknown auth type {self._auth_type}")

    def deploy_spec(self, model_spec: ModelSpec):

        api_client = self._get_api_client()
        env_api = EnvironmentApi(api_client)
        user = env_api.read_user()

        print(user)

        # TODO: Use KubernetesSpec().spec
        sd = SeldonDeployment(
            kind="SeldonDeployment",
            api_version="machinelearning.seldon.io/v1",
            metadata=ObjectMeta(
                name=model_spec.model_details.name, namespace=model_spec.runtime_options.k8s_options.namespace
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
                        replicas=model_spec.runtime_options.k8s_options.replicas,
                    )
                ]
            ),
        )

        dep_instance = SeldonDeploymentsApi(api_client)
        dep_instance.create_seldon_deployment(model_spec.runtime_options.k8s_options.namespace, sd)

    def wait_ready_spec(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        ready = False
        t0 = time.time()
        api_client = self._get_api_client()
        dep_instance = SeldonDeploymentsApi(api_client)
        while not ready:
            sdep: SeldonDeployment = dep_instance.read_seldon_deployment(
                model_spec.model_details.name, model_spec.runtime_options.k8s_options.namespace
            )
            sdep_dict = sdep.to_dict()
            ready = sdep_dict["status"]["state"] == "Available"
            if timeout_secs is not None:
                t1 = time.time()
                if t1 - t0 > timeout_secs:
                    return ready
        return ready

    def undeploy_spec(self, model_spec: ModelSpec):
        api_client = self._get_api_client()
        dep_instance = SeldonDeploymentsApi(api_client)
        dep_instance.delete_seldon_deployment(
            model_spec.model_details.name, model_spec.runtime_options.k8s_options.namespace, _preload_content=False
        )

    def get_endpoint_spec(self, model_spec: ModelSpec):
        endpoint = Endpoint()
        return endpoint.get_url(model_spec)

    def get_headers(self, model_details: ModelDetails) -> Dict[str, str]:
        return {}

    def remote(self, model_spec: ModelSpec, *args, **kwargs) -> Any:
        api_client = self._get_api_client()
        req = model_spec.protocol.to_protocol_request(*args, **kwargs)
        predict_instance = PredictApi(api_client)
        prediction = predict_instance.predict_seldon_deployment(
            model_spec.model_details.name, model_spec.runtime_options.k8s_options.namespace, prediction=req
        )
        return model_spec.protocol.from_protocol_response(prediction, model_spec.model_details.outputs)

    def to_k8s_yaml_spec(self, model_spec: ModelSpec) -> str:
        srt = SeldonKubernetesRuntime()
        return srt.to_k8s_yaml_spec(model_spec)
