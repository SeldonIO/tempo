import os

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from tempo.serve.constants import (
    ENV_K8S_SERVICE_HOST,
    DefaultInsightsImage,
    DefaultInsightsPort,
    DefaultInsightsServiceName,
    DefaultRedisImage,
    DefaultRedisPort,
    DefaultRedisServiceName,
    DefaultSeldonSystemNamespace,
)
from tempo.utils import logger


def create_k8s_client():
    inside_cluster = os.getenv(ENV_K8S_SERVICE_HOST)
    if inside_cluster:
        logger.debug("Loading cluster local config")
        config.load_incluster_config()
    else:
        logger.debug("Loading external kubernetes config")
        config.load_kube_config()


def deploy_insights_message_dumper():
    create_k8s_client()
    api_instance = client.CoreV1Api()

    try:
        api_instance.read_namespaced_pod(
            DefaultInsightsServiceName,
            DefaultSeldonSystemNamespace,
        )
        logger.debug(
            f"Pod with name {DefaultInsightsServiceName} in namespace {DefaultSeldonSystemNamespace} already exists"
        )
    except ApiException as e:
        if e.status != 404:
            raise e

        k8s_pod_spec = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": DefaultInsightsServiceName,
                "labels": {
                    "app": DefaultInsightsServiceName,
                },
            },
            "spec": {
                "containers": [
                    {
                        "name": "default",
                        "image": DefaultInsightsImage,
                        "ports": [{"containerPort": DefaultInsightsPort}],
                    }
                ]
            },
        }
        logger.debug(f"Creating kubernetes insights manager pod: {k8s_pod_spec}")
        api_instance.create_namespaced_pod(DefaultSeldonSystemNamespace, k8s_pod_spec)

    try:
        api_instance.read_namespaced_service(
            DefaultInsightsServiceName,
            DefaultSeldonSystemNamespace,
        )
        logger.debug(
            f"Service with name {DefaultInsightsServiceName} in namespace {DefaultSeldonSystemNamespace} already exists"
        )
    except ApiException as e:
        if e.status != 404:
            raise e

        k8s_svc_spec = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": DefaultInsightsServiceName,
            },
            "spec": {
                "selector": {"app": DefaultInsightsServiceName},
                "ports": [
                    {
                        "port": DefaultInsightsPort,
                        "targetPort": DefaultInsightsPort,
                    }
                ],
            },
        }

        logger.debug(f"Creating kubernetes insights manager pod: {k8s_svc_spec}")
        api_instance.create_namespaced_service(DefaultSeldonSystemNamespace, k8s_svc_spec)


def undeploy_insights_message_dumper():
    create_k8s_client()
    api_instance = client.CoreV1Api()
    try:
        api_instance.delete_namespaced_pod(DefaultInsightsServiceName, DefaultSeldonSystemNamespace)
    except ApiException as e:
        if e.status != 404:
            raise e

    try:
        api_instance.delete_namespaced_service(DefaultInsightsServiceName, DefaultSeldonSystemNamespace)
    except ApiException as e:
        if e.status != 404:
            raise e


def get_logs_insights_message_dumper():
    create_k8s_client()
    api_instance = client.CoreV1Api()
    response = api_instance.read_namespaced_pod_log(DefaultInsightsServiceName, DefaultSeldonSystemNamespace)
    return response


def deploy_redis():
    create_k8s_client()
    api_instance = client.CoreV1Api()

    try:
        api_instance.read_namespaced_pod(
            DefaultRedisServiceName,
            DefaultSeldonSystemNamespace,
        )
        logger.debug(
            f"Pod with name {DefaultRedisServiceName} in namespace {DefaultSeldonSystemNamespace} already exists"
        )
    except ApiException as e:
        if e.status != 404:
            raise e

        k8s_pod_spec = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": DefaultRedisServiceName,
                "labels": {
                    "app": DefaultRedisServiceName,
                },
            },
            "spec": {
                "containers": [
                    {
                        "name": "default",
                        "image": DefaultRedisImage,
                        "ports": [{"containerPort": DefaultRedisPort}],
                    }
                ]
            },
        }
        logger.debug(f"Creating kubernetes redis pod: {k8s_pod_spec}")
        api_instance.create_namespaced_pod(DefaultSeldonSystemNamespace, k8s_pod_spec)

    try:
        api_instance.read_namespaced_service(
            DefaultRedisServiceName,
            DefaultSeldonSystemNamespace,
        )
        logger.debug(
            f"Service with name {DefaultRedisServiceName} in namespace {DefaultSeldonSystemNamespace} already exists"
        )
    except ApiException as e:
        if e.status != 404:
            raise e

        k8s_svc_spec = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": DefaultRedisServiceName,
            },
            "spec": {
                "selector": {"app": DefaultRedisServiceName},
                "ports": [
                    {
                        "port": DefaultRedisPort,
                        "targetPort": DefaultRedisPort,
                    }
                ],
            },
        }

        logger.debug(f"Creating kubernetes redis pod: {k8s_svc_spec}")
        api_instance.create_namespaced_service(DefaultSeldonSystemNamespace, k8s_svc_spec)


def undeploy_redis():
    create_k8s_client()
    api_instance = client.CoreV1Api()
    try:
        api_instance.delete_namespaced_pod(DefaultRedisServiceName, DefaultSeldonSystemNamespace)
    except ApiException as e:
        if e.status != 404:
            raise e

    try:
        api_instance.delete_namespaced_service(DefaultRedisServiceName, DefaultSeldonSystemNamespace)
    except ApiException as e:
        if e.status != 404:
            raise e
