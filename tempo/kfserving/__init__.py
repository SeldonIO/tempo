from .protocol import KFServingV1Protocol, KFServingV2Protocol
from .k8s import KFServingKubernetesRuntime

__all__ = ["KFServingV1Protocol", "KFServingV2Protocol", "KFServingKubernetesRuntime"]
