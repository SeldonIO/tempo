import os

import docker

from tempo.docker.constants import DefaultNetworkName
from tempo.serve.constants import DefaultInsightsImage, DefaultInsightsPort, DefaultInsightsServiceName
from tempo.utils import logger


def create_network(docker_client: docker.client.DockerClient, network_name=DefaultNetworkName):
    try:
        docker_client.networks.get(network_id=network_name)
    except docker.errors.NotFound:
        docker_client.networks.create(name=DefaultNetworkName)


def deploy_insights_message_dumper(
    name=DefaultInsightsServiceName, image=DefaultInsightsImage, port=DefaultInsightsPort
):
    docker_client = docker.from_env()
    try:
        docker_client.containers.get(DefaultInsightsServiceName)
    except docker.errors.NotFound:
        pass
    else:
        logger.info("Attempted to deploy message dumper but already deployed")
        return
    uid = os.getuid()
    create_network(docker_client)
    docker_client.containers.run(
        name=name,
        ports={f"{port}/tcp": DefaultInsightsPort},
        image=image,
        detach=True,
        network=DefaultNetworkName,
        user=uid,
    )


def undeploy_insights_message_dumper(name=DefaultInsightsServiceName):
    docker_client = docker.from_env()
    # TODO: Get from constant
    try:
        container = docker_client.containers.get(name)
    except docker.errors.NotFound:
        logger.info("Attempted to undeploy insights dumper but container not running")
        return
    container.remove(force=True)


def get_logs_insights_message_dumper(name=DefaultInsightsServiceName):
    docker_client = docker.from_env()
    # TODO: Get from constant
    try:
        container = docker_client.containers.get(name)
    except docker.errors.NotFound:
        logger.info("Attempted to undeploy insights dumper but container not running")
        return
    return container.logs().decode("utf-8")
