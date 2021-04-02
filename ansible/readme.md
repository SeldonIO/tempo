## Configuring Kind Cluster with Ansible

Note: this is in a very early stage of exploration.

Requirements:
```bash
pip install ansible openshift
ansible-galaxy collection install community.kubernetes community.docker
```

**WARNING:** Make sure your `kubectl` is configured against a local `kind` cluster!!

Unless you use `kind` role (part of default.yaml) as this one will create a `kind` cluster for you and configure the `kubectl`.

Basic usage:
```bash
╭─rskolasinski at machine42 [~/w/s/k8s_ansible] (master ✚1…2) (14:23:36)                                                       (⎈ n/a|default)
╰λ ansible-playbook playbooks/default.yaml

PLAY [Install Seldon Core] *******************************************************************************************************************

TASK [Gathering Facts] ***********************************************************************************************************************
ok: [localhost]

TASK [kind : Check if KinD Cluster already exists: ansible] **********************************************************************************
ok: [localhost]

TASK [kind : Start KinD Cluster: 'ansible'] **************************************************************************************************
changed: [localhost]

TASK [kind : Echo message about KinD creation output] ****************************************************************************************
ok: [localhost] => {
    "msg": [
        "Creating cluster \"ansible\" ...",
        " • Ensuring node image (kindest/node:v1.18.15) 🖼  ...",
        " ✓ Ensuring node image (kindest/node:v1.18.15) 🖼",
        " • Preparing nodes 📦   ...",
        " ✓ Preparing nodes 📦 ",
        " • Writing configuration 📜  ...",
        " ✓ Writing configuration 📜",
        " • Starting control-plane 🕹️  ...",
        " ✓ Starting control-plane 🕹️",
        " • Installing CNI 🔌  ...",
        " ✓ Installing CNI 🔌",
        " • Installing StorageClass 💾  ...",
        " ✓ Installing StorageClass 💾",
        "Set kubectl context to \"kind-ansible\"",
        "You can now use your cluster with:",
        "",
        "kubectl cluster-info --context kind-ansible",
        "",
        "Not sure what to do next? 😅  Check out https://kind.sigs.k8s.io/docs/user/quick-start/"
    ]
}

TASK [kind : Export kubeconfig for KinD Cluster: 'ansible'] **********************************************************************************
changed: [localhost]

TASK [kind : Create a k8s namespace: seldon] *************************************************************************************************
changed: [localhost]

TASK [kind : Set default for kubectl namespace: seldon] **************************************************************************************
changed: [localhost]

TASK [ambassador : Create a k8s namespace: ambassador] ***************************************************************************************
changed: [localhost]

TASK [ambassador : Add ambassador chart repo] ************************************************************************************************
ok: [localhost]

TASK [ambassador : Install ambassador] *******************************************************************************************************
changed: [localhost]

TASK [istio : Check if Istio 1.7.6 already downloaded.] **************************************************************************************
ok: [localhost]

TASK [istio : Create .resources directory if does not exist: /home/rskolasinski/work/seldondev-utils/k8s_ansible/ansible-tools/.resources/] ***
ok: [localhost]

TASK [istio : Download Istio 1.7.6] **********************************************************************************************************
skipping: [localhost]

TASK [istio : Install Istio 1.7.6] ***********************************************************************************************************
changed: [localhost]

TASK [istio : Verify Install Istio 1.7.6] ****************************************************************************************************
skipping: [localhost]

TASK [istio : Create Seldon Gateway] *********************************************************************************************************
changed: [localhost]

TASK [minio : Create a k8s namespace: minio-system] ******************************************************************************************
changed: [localhost]

TASK [minio : Add minio chart repo] **********************************************************************************************************
ok: [localhost]

TASK [minio : Install MinIO] *****************************************************************************************************************
changed: [localhost]

TASK [minio : Echo message about client configuration] ***************************************************************************************
ok: [localhost] => {
    "msg": [
        "MinIO installed in the cluster. To configure your client launch",
        "kubectl port-forward -n minio-system svc/minio 8090:9000",
        "in one terminal and execute",
        "mc config host add minio-seldon http://localhost:8090 ACCESS_KEY SECRET_KEY"
    ]
}

TASK [seldon_core : Create a k8s namespaces] *************************************************************************************************
changed: [localhost] => (item=seldon-system)
ok: [localhost] => (item=default)
ok: [localhost] => (item=seldon)
changed: [localhost] => (item=production)

TASK [seldon_core : Create MinIO Secret] *****************************************************************************************************
changed: [localhost] => (item=default)
changed: [localhost] => (item=seldon)
changed: [localhost] => (item=production)

TASK [seldon_core : Git clone Seldon Core repo and checkout v1.7.0] **************************************************************************
ok: [localhost]

TASK [seldon_core : Set Seldon Core Directory] ***********************************************************************************************
ok: [localhost]

TASK [seldon_core : Deploy Seldon Core] ******************************************************************************************************
changed: [localhost]

PLAY RECAP ***********************************************************************************************************************************
localhost                  : ok=23   changed=13   unreachable=0    failed=0    skipped=2    rescued=0    ignored=0
