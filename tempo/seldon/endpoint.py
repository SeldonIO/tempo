import os

from kubernetes import client, config

from tempo.serve.ingress import create_ingress
from tempo.serve.runtime import ModelSpec
from tempo.utils import logger

ENV_K8S_SERVICE_HOST = "KUBERNETES_SERVICE_HOST"
ISTIO_GATEWAY = "istio"


class Endpoint(object):
    """A Model Endpoint

    Only handles istio and seldon at present.

    """

    def __init__(self):
        self.inside_cluster = os.getenv(ENV_K8S_SERVICE_HOST)
        try:
            if self.inside_cluster:
                logger.debug("Loading cluster local config")
                config.load_incluster_config()
            else:
                logger.debug("Loading external kubernetes config")
                config.load_kube_config()
        except Exception:
            logger.warning("Failed to load kubeconfig. Only local mode is possible.")

    def get_url(self, model_spec: ModelSpec):
        if self.inside_cluster is None:
            ingress = create_ingress(model_spec)
            ingress_host_url = ingress.get_external_host_url(model_spec)
            return (
                f"{ingress_host_url}"
                + f"/seldon/{model_spec.runtime_options.k8s_options.namespace}/"
                + f"{model_spec.model_details.name}"
                + model_spec.protocol.get_predict_path(model_spec.model_details)
            )
        else:
            # TODO check why needed this here
            config.load_incluster_config()
            api_instance = client.CustomObjectsApi()
            api_response = api_instance.get_namespaced_custom_object_status(
                "machinelearning.seldon.io",
                "v1",
                model_spec.runtime_options.k8s_options.namespace,
                "seldondeployments",
                model_spec.model_details.name,
            )
            return api_response["status"]["address"]["url"]
