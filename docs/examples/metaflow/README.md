# Tempo Metaflow Example

This example assumes you have [Metaflow](https://metaflow.org/) installed and have a production setup of metaflow on AWS active. It also assumes a production Kubernetes cluster running Seldon Core.

You [Ansible to install Seldon Core on a Kubernetes cluster](https://github.com/SeldonIO/ansible-k8s-collection).

## Requirements

Install metaflow.

```
pip install metaflow
```

## Kubernetes Authentication from AWS Metaflow Batch

For Metaflow AWS batch jobs to run tempo commands against a remote Kubernetes cluster we will need to pass authentication to the flow.

### GKE

For GKE we will need to create two files in the flow src folder:

```bash
kubeconfig.yaml
gsa-key.json
```

Follow the steps outlined in [GKE server authentication](https://cloud.google.com/kubernetes-engine/docs/how-to/api-server-authentication#environments-without-gcloud).


```
kubectl create ns production
```

```
kubectl create -f k8s/tempo-pipeline-rbac.yaml -n production
```

From the provided template in the k8s folder create your s3 secret yaml.

```
apiVersion: v1
kind: Secret
metadata:
  name: s3-secret
type: Opaque
stringData:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: aws
  RCLONE_CONFIG_S3_ENV_AUTH: "false"
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: <key>
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: <secret>
```

Create your s3 credentials secret.

```
kubectl create -f k8s/s3_secret.yaml -n production
```

Run the pipeline on AWS with a custom image which has conda installed.

```bash
python src/irisflow.py --environment=conda --with batch:image=seldonio/seldon-core-s2i-python37-ubi8:1.10.0-dev run
```

The Pipeline has various stages:

 1. Download Iris data
 2. Train SKLearn and XGBoost models.
 3. Deploy a Tempo pipeline including your two models and custom logic to orchestrate them.