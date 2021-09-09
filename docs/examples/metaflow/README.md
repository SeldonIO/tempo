# End to End ML with Metaflow and Tempo

We will train two models and deploy them with tempo within a Metaflow pipeline. To understand the core example see [here](https://tempo.readthedocs.io/en/latest/examples/multi-model/README.html)

![archtecture](architecture.png)

## MetaFlow Prequisites


### Install metaflow locally

```
pip install metaflow
```

### Setup Conda-Forge Support

The flow will use conda-forge so you need to add that channel to conda.

```
conda config --add channels conda-forge
```



## Iris Flow Summary


```python
!python src/irisflow.py --environment=conda show
```

## Run Flow locally to deploy to Docker

To run the workflow with a local Docker deployment use the flag:

```
--tempo-on-docker true
```



```python
!python src/irisflow.py --environment=conda run 
```

## Make Predictions with Metaflow Tempo Artifact


```python
from metaflow import Flow
import numpy as np
run = Flow('IrisFlow').latest_run
client = run.data.client_model
client.predict(np.array([[1, 2, 3, 4]]))
```

## Run Flow on AWS and Deploy to Remote Kubernetes

We will now run our flow on AWS Batch and will launch Tempo artifacts onto a remote Kubernetes cluster. 

### Setup AWS Metaflow Support

Note at present this is required even for a local run as artifacts are stored on S3.

[Install Metaflow with remote AWS support](https://docs.metaflow.org/metaflow-on-aws/metaflow-on-aws).

### Seldon Requirements

For deploying to a remote Kubernetes cluster with Seldon Core installed do the following steps:

#### Install Seldon Core on your Kubernetes Cluster

Create a GKE cluster and install Seldon Core on it using [Ansible to install Seldon Core on a Kubernetes cluster](https://github.com/SeldonIO/ansible-k8s-collection).


### K8S Auth from Metaflow

To deploy services to our Kubernetes cluster with Seldon Core installed, Metaflow steps that run on AWS Batch and use tempo will need to be able to access K8S API. This step will depend on whether you're using GKE or AWS EKS to run 
your cluster.

#### Option 1. K8S cluster runs on GKE

We will need to create two files in the flow src folder:

```bash
kubeconfig.yaml
gsa-key.json
```

Follow the steps outlined in [GKE server authentication](https://cloud.google.com/kubernetes-engine/docs/how-to/api-server-authentication#environments-without-gcloud).




#### Option 2. K8S cluster runs on AWS EKS

Make note of two AWS IAM role names, for example find them in the IAM console. The names depend on how you deployed Metaflow and EKS in the first place:

1. The role used by Metaflow tasks executed on AWS Batch. If you used the default CloudFormation template to deploy Metaflow, it is the role that has `*BatchS3TaskRole*` in its name.

2. The role used by EKS nodes. If you used `eksctl` to create your EKS cluster, it is the role that starts with `eksctl-<your-cluster-name>-NodeInstanceRole-*`

Now, we need to make sure that AWS Batch role has permissions to access the K8S cluster. For this, add a policy to the AWS Batch task role(1) that has `eks:*` permissions on your EKS cluster (TODO: narrow this down).

You'll also need to add a mapping for that role to `aws-auth` ConfigMap in `kube-system` namespace. For more details, see [AWS docs](https://docs.aws.amazon.com/eks/latest/userguide/add-user-role.html) (under "To add an IAM user or role to an Amazon EKS cluster"). In short, you'd need to add this to `mapRoles` section in the aws-auth ConfigMap:
```
     - rolearn: <batch task role ARN>
       username: cluster-admin
       groups:
         - system:masters
```

We also need to make sure that the code running in K8S can access S3. For this, add a policy to the EKS node role (2) to allow it to read and write Metaflow S3 buckets.

### S3 Authentication
Services deployed to Seldon will need to access Metaflow S3 bucket to download trained models. The exact configuration will depend on whether you're using GKE or AWS EKS to run your cluster.

From the base templates provided below, create your `k8s/s3_secret.yaml`.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: s3-secret
type: Opaque
stringData:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: aws
  RCLONE_CONFIG_S3_BUCKET_REGION: <region>
  <...cloud-dependent s3 auth settings (see below)>
```

For GKE, to access S3 we'll need to add the following variables to use key/secret auth:
```yaml
  RCLONE_CONFIG_S3_ENV_AUTH: "false"
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: <key>
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: <secret>
```

For AWS EKS, we'll use the instance role assigned to the node, we'll only need to set one env variable:
```yaml
RCLONE_CONFIG_S3_ENV_AUTH: "true"
```

We provide two templates to use in the `k8s` folder:

```
s3_secret.yaml.tmpl.aws
s3_secret.yaml.tmpl.gke
```

Use one to create the file `s3_secret.yaml` in the same folder


## Setup RBAC and Secret on Kubernetes Cluster

These steps assume you have authenticated to your cluster with kubectl configuration


```python
!kubectl create ns production
```


```python
!kubectl create -f k8s/tempo-pipeline-rbac.yaml -n production
```

Create a Secret from the `k8s/s3_secret.yaml.tmpl` file by adding your AWS Key that can read from S3 and saving as `k8s/s3_secret.yaml`


```python
!kubectl create -f k8s/s3_secret.yaml -n production
```

## Run Metaflow on AWS Batch


```python
!python src/irisflow.py \
    --environment=conda \
    --with batch:image=seldonio/seldon-core-s2i-python37-ubi8:1.10.0-dev \
    run
```

## Make Predictions with Metaflow Tempo Artifact


```python
from metaflow import Flow
run = Flow('IrisFlow').latest_run
client = run.data.client_model
import numpy as np
client.predict(np.array([[1, 2, 3, 4]]))
```


```python

```
