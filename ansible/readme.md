# Ansible Playbooks


## Requirements

Requirements:
```bash
pip install ansible openshift
ansible-galaxy collection install git+https://github.com/SeldonIO/ansible-k8s-collection.git,v0.1.0
```

## Playbooks

Note: the optional `-e skip_kind=1` flag allow to skip creation of Kind cluster


### playbooks/seldon_core.yaml

 * Create a kind cluster
 * Install metallb
 * Install istio
 * Install Minio
 * Install Seldon Core

```bash
ansible-playbook playbooks/seldon_core.yaml [-e skip_kind=1]
```

### playbooks/kfserving.yaml

 * Create a Kind cluster
 * Install metallb
 * Install istio
 * Install Knative Serving
 * Install Minio
 * Install certmanager
 * Install kfserving

```bash
ansible-playbook playbooks/kfserving.yaml [-e skip_kind=1]
```
