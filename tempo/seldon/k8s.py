import os
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from tempo.seldon.endpoint import Endpoint
from tempo.seldon.protocol import SeldonProtocol
from tempo.serve.runtime import Runtime
from tempo.utils import logger
from tempo.serve.metadata import ModelDetails, ModelFramework

ENV_K8S_SERVICE_HOST = "KUBERNETES_SERVICE_HOST"


class SeldonKubernetesRuntime(Runtime):

    Implementations = {
        ModelFramework.SKLearn: "SKLEARN_SERVER",
        ModelFramework.XGBoost: "XGBOOST_SERVER",
        ModelFramework.MLFlow: "MLFLOW_SERVER",
        ModelFramework.Tensorflow: "TENSORFLOW_SERVER",
        ModelFramework.Triton: "TRITON_SERVER"
    }

    def __init__(self, namespace="default", replicas=1, protocol = None):
        self.namespace = namespace
        self.replicas = replicas
        self.create_k8s_client()
        if protocol is None:
            self.protocol = SeldonProtocol()
        else:
            self.protocol = protocol

    def get_protocol(self):
        return SeldonProtocol()

    def create_k8s_client(self):
        inside_cluster = os.getenv(ENV_K8S_SERVICE_HOST)
        if inside_cluster:
            logger.debug("Loading cluster local config")
            config.load_incluster_config()
        else:
            logger.debug("Loading external kubernetes config")
            config.load_kube_config()

    def get_endpoint(self, model_details: ModelDetails):
        endpoint = Endpoint(model_details.name, self.namespace, SeldonProtocol())
        return endpoint.get_url()

    def undeploy(self, model_details: ModelDetails):
        api_instance = client.CustomObjectsApi()
        api_instance.delete_namespaced_custom_object(
            "machinelearning.seldon.io",
            "v1",
            self.namespace,
            "seldondeployments",
            model_details.name,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )

    def deploy(self, model_details: ModelDetails):
        model_spec = self._get_spec(model_details)
        logger.debug(model_spec)

        api_instance = client.CustomObjectsApi()

        try:
            existing = api_instance.get_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                self.namespace,
                "seldondeployments",
                model_details.name,
            )
            model_spec["metadata"]["resourceVersion"] = existing["metadata"][
                "resourceVersion"
            ]
            api_instance.replace_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                self.namespace,
                "seldondeployments",
                model_details.name,
                model_spec,
            )
        except ApiException as e:
            if e.status == 404:
                api_instance.create_namespaced_custom_object(
                    "machinelearning.seldon.io",
                    "v1",
                    self.namespace,
                    "seldondeployments",
                    model_spec,
                )
            else:
                raise e

    def _get_spec(self, model_details: ModelDetails) -> dict:
        model_implementation = self.Implementations[model_details.platform]
        return {
            "apiVersion": "machinelearning.seldon.io/v1",
            "kind": "SeldonDeployment",
            "metadata": {"name": model_details.name, "namespace": self.namespace},
            "spec": {
                "predictors": [
                    {
                        "graph": {
                            "implementation": model_implementation,
                            "modelUri": model_details.uri,
                            "name": "classifier",
                        },
                        "name": "default",
                        "replicas": self.replicas,
                    }
                ]
            },
        }

    def to_k8s_yaml(self, model_details: ModelDetails) -> str:
        d = self._get_spec(model_details)
        return yaml.safe_dump(d)