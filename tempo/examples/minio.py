from kubernetes import client, config

from tempo.conf import settings

RCloneConf = """
[s3]
type = s3
provider = minio
env_auth = false
access_key_id = minioadmin
secret_access_key = minioadmin
endpoint = http://{MINIO_IP}:9000
"""


def create_minio_rclone(path: str):
    config.load_kube_config()
    api_instance = client.CoreV1Api()
    res = api_instance.list_namespaced_service("minio-system", field_selector="metadata.name=minio")
    minio_ip = res.items[0].status.load_balancer.ingress[0].ip
    rclone_conf = RCloneConf.replace("{MINIO_IP}", minio_ip)
    with open(path, "w") as f:
        f.write(rclone_conf)
    settings.rclone_cfg = path
