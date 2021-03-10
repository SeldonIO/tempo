import time
from enum import Enum
from http import cookies
from pathlib import Path
from typing import Any, Dict

from seldon_deploy_sdk import ApiClient, Configuration, PredictApi, SeldonDeploymentsApi
from seldon_deploy_sdk.auth import SessionAuthenticator
from seldon_deploy_sdk.models.object_meta import ObjectMeta
from seldon_deploy_sdk.models.predictive_unit import PredictiveUnit
from seldon_deploy_sdk.models.predictor_spec import PredictorSpec
from seldon_deploy_sdk.models.seldon_deployment import SeldonDeployment
from seldon_deploy_sdk.models.seldon_deployment_spec import SeldonDeploymentSpec

from tempo.seldon.endpoint import Endpoint
from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.seldon.protocol import SeldonProtocol
from tempo.seldon.specs import KubernetesSpec
from tempo.serve.metadata import KubernetesOptions, ModelDetails
from tempo.serve.runtime import Runtime


class SeldonDeployAuthType(Enum):
    session_cookie = "session_cookie"


class SeldonDeployRuntime(Runtime):
    def __init__(
        self,
        host: str = None,
        user: str = None,
        password: str = None,
        auth_type: SeldonDeployAuthType = SeldonDeployAuthType.session_cookie,
        k8s_options: KubernetesOptions = None,
        _protocol=None,
    ):
        if k8s_options is None:
            k8s_options = KubernetesOptions()
        self._k8s_options = k8s_options
        # Load config if needed
        sd_config_values = {}
        if host is None or user is None or password is None:
            with open(str(Path.home()) + "/.config/seldon/seldon-deploy/sdconfig.txt") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    key, value = line.strip().split("=", 1)
                    sd_config_values[key] = value
        if host is None:
            self._host = (
                sd_config_values["EXTERNAL_PROTOCOL"]
                + "://"
                + sd_config_values["EXTERNAL_HOST"]
                + "/seldon-deploy/api/v1alpha1"
            )
        else:
            self._host = host
        if user is None:
            self._user = sd_config_values["SD_USER_EMAIL"]
        else:
            self._user = user
        if password is None:
            self._password = sd_config_values["SD_PASSWORD"]
        else:
            self._password = password
        self._auth_type = auth_type
        if _protocol is None:
            self.protocol = SeldonProtocol()
        else:
            self.protocol = _protocol

    def _get_api_client(self):
        config = Configuration()
        config.host = self._host
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
        else:
            raise ValueError(f"Unknown auth type {self._auth_type}")

    def deploy(self, model_details: ModelDetails):
        # TODO: Use KubernetesSpec().spec
        sd = SeldonDeployment(
            kind="SeldonDeployment",
            api_version="machinelearning.seldon.io/v1",
            metadata=ObjectMeta(name=model_details.name, namespace=self._k8s_options.namespace),
            spec=SeldonDeploymentSpec(
                predictors=[
                    PredictorSpec(
                        graph=PredictiveUnit(
                            implementation=KubernetesSpec.Implementations[model_details.platform],
                            model_uri=model_details.uri,
                            name="model",
                        ),
                        name="default",
                        replicas=self._k8s_options.replicas,
                    )
                ]
            ),
        )
        api_client = self._get_api_client()
        dep_instance = SeldonDeploymentsApi(api_client)
        dep_instance.create_seldon_deployment(self._k8s_options.namespace, sd)

    def wait_ready(self, model_details: ModelDetails, timeout_secs=None) -> bool:
        ready = False
        t0 = time.time()
        api_client = self._get_api_client()
        dep_instance = SeldonDeploymentsApi(api_client)
        while not ready:
            sdep: SeldonDeployment = dep_instance.read_seldon_deployment(
                model_details.name, self._k8s_options.namespace
            )
            sdep_dict = sdep.to_dict()
            ready = sdep_dict["status"]["state"] == "Available"
            if timeout_secs is not None:
                t1 = time.time()
                if t1 - t0 > timeout_secs:
                    return ready
        return ready

    def undeploy(self, model_details: ModelDetails):
        api_client = self._get_api_client()
        dep_instance = SeldonDeploymentsApi(api_client)
        dep_instance.delete_seldon_deployment(model_details.name, self._k8s_options.namespace, _preload_content=False)

    def get_endpoint(self, model_details: ModelDetails):
        endpoint = Endpoint(model_details.name, self._k8s_options.namespace, self.protocol)
        return endpoint.get_url(model_details)

    def get_headers(self, model_details: ModelDetails) -> Dict[str, str]:
        return {}

    def remote(self, model_details: ModelDetails, *args, **kwargs) -> Any:
        api_client = self._get_api_client()
        protocol = self.get_protocol()
        req = protocol.to_protocol_request(*args, **kwargs)
        predict_instance = PredictApi(api_client)
        prediction = predict_instance.predict_seldon_deployment(
            model_details.name, self._k8s_options.namespace, prediction=req
        )
        return protocol.from_protocol_response(prediction, model_details.outputs)

    def get_protocol(self):
        return self.protocol

    def to_k8s_yaml(self, model_details: ModelDetails) -> str:
        srt = SeldonKubernetesRuntime(k8s_options=self._k8s_options, protocol=self.protocol)
        return srt.to_k8s_yaml(model_details)
