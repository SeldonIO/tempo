# Ansible Playbooks


## Requirements

Requirements:
```bash
pip install ansible openshift
ansible-galaxy collection install community.kubernetes community.docker
```

## Playbooks

### playbooks/default.yaml

 * Create a kind cluster
 * Install metallb
 * Install istio
 * Install Minio
 * Install Seldon Core

```
ansible-playbook playbooks/default.yaml
```

### playbooks/core_istio.yaml

 * Install istio
 * Install Seldon Core

```
ansible-playbook playbooks/core_istio.yaml
```

### playbooks/kfserving.yaml

 * Create a Kind cluster
 * Install metallb
 * Install istio
 * Install Knative Serving
 * Install Minio
 * Install certmanager
 * Install kfserving

```
ansible-playbook playbooks/kfserving.yaml
```

### playbooks/core_knative.yaml

 * Create a kind cluster
 * Install metallb
 * Install istio
 * Install Knative Serving
 * Install Kntive Eventing
 * Install Minio
 * Install Seldon Core

```
ansible-playbook playbooks/core_knative.yaml
```
