import json
import os
import time
from typing import Any, Dict, Optional

import requests
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from tempo.kfserving.endpoint import Endpoint
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.seldon.constants import MLSERVER_IMAGE
from tempo.seldon.specs import DefaultModelsPath, DefaultServiceAccountName
from tempo.serve.constants import ENV_TEMPO_RUNTIME_OPTIONS
from tempo.serve.metadata import ModelFramework, RuntimeOptions
from tempo.serve.remote import Remote
from tempo.serve.runtime import ModelSpec, Runtime
from tempo.utils import logger

DefaultHTTPPort = "8080"
DefaultGRPCPort = "9000"

ENV_K8S_SERVICE_HOST = "KUBERNETES_SERVICE_HOST"

Implementations = {
    ModelFramework.SKLearn: "sklearn",
    ModelFramework.XGBoost: "xgboost",
    ModelFramework.Tensorflow: "tensorflow",
    ModelFramework.PyTorch: "triton",
    ModelFramework.ONNX: "triton",
    ModelFramework.TensorRT: "triton",
}


class KFServingKubernetesRuntime(Runtime, Remote):
    def __init__(self, runtime_options: Optional[RuntimeOptions] = None):
        if runtime_options:
            runtime_options.runtime = "tempo.kfserving.KFServingKubernetesRuntime"
        super().__init__(runtime_options)

    def _inside_cluster(self):
        return os.getenv(ENV_K8S_SERVICE_HOST)

    def create_k8s_client(self):
        if self._inside_cluster():
            logger.debug("Loading cluster local config")
            config.load_incluster_config()
            return True
        else:
            logger.debug("Loading external kubernetes config")
            config.load_kube_config()
            return False

    def get_endpoint_spec(self, model_spec: ModelSpec) -> str:
        endpoint = Endpoint()
        return endpoint.get_url(model_spec)

    def get_headers(self, model_spec: ModelSpec) -> Dict[str, str]:
        if not self._inside_cluster():
            endpoint = Endpoint()
            service_host = endpoint.get_service_host(model_spec)
            return {"Host": service_host}
        else:
            return {}

    def remote(self, model_spec: ModelSpec, *args, **kwargs) -> Any:
        req = model_spec.protocol.to_protocol_request(*args, **kwargs)
        endpoint = self.get_endpoint_spec(model_spec)
        logger.debug("Endpoint is", endpoint)
        headers = self.get_headers(model_spec)
        logger.debug("Headers are", headers)
        response_raw = requests.post(
            endpoint, json=req, headers=headers, verify=model_spec.runtime_options.ingress_options.verify_ssl
        )
        if response_raw.status_code == 200:
            return model_spec.protocol.from_protocol_response(response_raw.json(), model_spec.model_details.outputs)
        else:
            raise ValueError("Bad return code", response_raw.status_code, response_raw.text)

    def undeploy_spec(self, model_spec: ModelSpec):
        self.create_k8s_client()
        api_instance = client.CustomObjectsApi()
        api_instance.delete_namespaced_custom_object(
            "serving.kubeflow.org",
            "v1beta1",
            model_spec.runtime_options.k8s_options.namespace,
            "inferenceservices",
            model_spec.model_details.name,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )

    def deploy_spec(self, model_spec: ModelSpec):
        spec = self._get_spec(model_spec)
        logger.debug(model_spec)
        self.create_k8s_client()
        api_instance = client.CustomObjectsApi()

        try:
            existing = api_instance.get_namespaced_custom_object(
                "serving.kubeflow.org",
                "v1beta1",
                model_spec.runtime_options.k8s_options.namespace,
                "inferenceservices",
                model_spec.model_details.name,
            )
            spec["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
            api_instance.replace_namespaced_custom_object(
                "serving.kubeflow.org",
                "v1beta1",
                model_spec.runtime_options.k8s_options.namespace,
                "inferenceservices",
                model_spec.model_details.name,
                spec,
            )
        except ApiException as e:
            if e.status == 404:
                api_instance.create_namespaced_custom_object(
                    "serving.kubeflow.org",
                    "v1beta1",
                    model_spec.runtime_options.k8s_options.namespace,
                    "inferenceservices",
                    spec,
                )
            else:
                raise e

    def wait_ready_spec(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        self.create_k8s_client()
        ready = False
        t0 = time.time()
        while not ready:
            api_instance = client.CustomObjectsApi()
            existing = api_instance.get_namespaced_custom_object(
                "serving.kubeflow.org",
                "v1beta1",
                model_spec.runtime_options.k8s_options.namespace,
                "inferenceservices",
                model_spec.model_details.name,
            )
            default_ready = False
            routes_ready = False
            conf_ready = False
            ingress_ready = False
            if "status" in existing and "conditions" in existing["status"]:
                for item in existing["status"]["conditions"]:
                    if item["type"] == "PredictorReady":
                        default_ready = item["status"] == "True"
                    elif item["type"] == "PredictorRouteReady":
                        routes_ready = item["status"] == "True"
                    elif item["type"] == "IngressReady":
                        ingress_ready = item["status"] == "True"
                    elif item["type"] == "PredictorConfigurationReady":
                        conf_ready = item["status"] == "True"
                ready = default_ready and routes_ready and ingress_ready and conf_ready
            if timeout_secs is not None:
                t1 = time.time()
                if t1 - t0 > timeout_secs:
                    return ready
            time.sleep(1)
        return ready

    def _get_spec(self, model_spec: ModelSpec) -> dict:
        if model_spec.model_details.platform == ModelFramework.TempoPipeline:
            serviceAccountName = model_spec.runtime_options.k8s_options.serviceAccountName
            if serviceAccountName is None:
                serviceAccountName = DefaultServiceAccountName
            return {
                "apiVersion": "serving.kubeflow.org/v1beta1",
                "kind": "InferenceService",
                "metadata": {
                    "name": model_spec.model_details.name,
                    "namespace": model_spec.runtime_options.k8s_options.namespace,
                },
                "spec": {
                    "predictor": {
                        "serviceAccountName": serviceAccountName,
                        "containers": [
                            {
                                "image": MLSERVER_IMAGE,
                                "name": "mlserver",
                                "env": [
                                    {
                                        "name": "STORAGE_URI",
                                        "value": model_spec.model_details.uri,
                                    },
                                    {
                                        "name": "MLSERVER_HTTP_PORT",
                                        "value": DefaultHTTPPort,
                                    },
                                    {
                                        "name": "MLSERVER_GRPC_PORT",
                                        "value": DefaultGRPCPort,
                                    },
                                    {
                                        "name": "MLSERVER_MODEL_IMPLEMENTATION",
                                        "value": "tempo.mlserver.InferenceRuntime",
                                    },
                                    {
                                        "name": "MLSERVER_MODEL_NAME",
                                        "value": model_spec.model_details.name,
                                    },
                                    {
                                        "name": "MLSERVER_MODEL_URI",
                                        "value": DefaultModelsPath,
                                    },
                                    {
                                        "name": ENV_TEMPO_RUNTIME_OPTIONS,
                                        "value": json.dumps(model_spec.runtime_options.dict()),
                                    },
                                ],
                            },
                        ],
                    },
                },
            }
        elif model_spec.model_details.platform in Implementations:
            model_implementation = Implementations[model_spec.model_details.platform]
            spec: Dict = {
                "apiVersion": "serving.kubeflow.org/v1beta1",
                "kind": "InferenceService",
                "metadata": {
                    "name": model_spec.model_details.name,
                    "namespace": model_spec.runtime_options.k8s_options.namespace,
                },
                "spec": {
                    "predictor": {
                        model_implementation: {"storageUri": model_spec.model_details.uri},
                    },
                },
            }
            if model_spec.runtime_options.k8s_options.serviceAccountName is not None:
                spec["spec"]["predictor"][
                    "serviceAccountName"
                ] = model_spec.runtime_options.k8s_options.serviceAccountName
            if isinstance(model_spec.protocol, KFServingV2Protocol):
                spec["spec"]["predictor"][model_implementation]["protocolVersion"] = "v2"
            return spec
        else:
            raise ValueError("Can't create spec for implementation ", model_spec.model_details.platform)

    def to_k8s_yaml_spec(self, model_spec: ModelSpec) -> str:
        d = self._get_spec(model_spec)
        return yaml.safe_dump(d)
