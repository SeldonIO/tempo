# Quickstart

## Installation
The latest Tempo release can be installed from [PyPI](https://pypi.org/project/mlops-tempo/):
```
pip install mlops-tempo
```
Alternatively, the development version can be installed:
```
pip install git+https://github.com/SeldonIO/tempo.git
```

## Tempo Prerequisites


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

## Kubernetes Runtime Prequisites

We provide a set of Ansible playbooks to create reproducible Kubernetes clusters with Kind for the demos.

These playbooks depend on Ansible roles that we publish in  [SeldonIO/ansible-k8s-collection](https://github.com/SeldonIO/ansible-k8s-collection) repository.

To obtain required ansible tools:
```bash
pip install ansible openshift
ansible-galaxy collection install git+https://github.com/SeldonIO/ansible-k8s-collection.git,v0.2.0
```

### Kubernetes Cluster with Seldon Core

To create a Kind cluster with istio, Seldon-Core and Minio run:
```bash
cd ansible
ansible-playbook playbooks/seldon_core.yaml
```

## Next Step

Create the `tempo-examples` conda environment and try the [introductory example](../examples/custom-model/README.html)
