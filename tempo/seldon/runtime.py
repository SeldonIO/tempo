from tempo.serve.metadata import RuntimeOptions


class SeldonCoreOptions(RuntimeOptions):
    runtime: str = "tempo.seldon.SeldonKubernetesRuntime"
    add_svc_orchestrator: bool = False
