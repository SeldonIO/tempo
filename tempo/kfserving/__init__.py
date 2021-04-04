from .k8s import KFServingKubernetesRuntime
from .protocol import KFServingV1Protocol, KFServingV2Protocol

__all__ = ["KFServingV1Protocol", "KFServingV2Protocol", "KFServingKubernetesRuntime"]
