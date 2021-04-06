# Quickstart

## Tempo Prequisites


 * [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
    * We use conda to provide reproducible environments for saving pipelines and also running demos.
 * [rclone](https://rclone.org/install/)
    * We use rclone to upload and download model artifacts to a wide range of storage systems.
 * [ansible](https://www.ansible.com/)
    * We use ansible to provide reproducible Kubernetes environments for the demos.

## Examples Conda Environment

We have created a conda environment to run all examples provided.

To create the environment run:

```
conda env create --name tempo-examples --file conda/tempo-examples.yaml
```

## Docker Runtime Prerequisites

Install [Docker](https://www.docker.com/) to run with the Docker runtime.

## Kubernetes Runtime prequisites

We provide Ansible playbooks to create reproducible Kubernetes clusters with Kind for the demos.


### Kubernetes Cluster with Seldon Core

To create a Kind cluster with istio, Seldon-Core and Minio run:

```
ansible-playbook ansible/playbooks/default.yaml
```

### Kubernetes Cluster with KFServing


To create a Kind cluster with istio, knative serving, KFServing and Minio run:


```
ansible-playbook ansible/playbooks/kfserving.yaml
```


## Next Step

Create the `tempo-examples` conda environment and try the [introductory example](../examples/intro/local.html)