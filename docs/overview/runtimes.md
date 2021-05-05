# Runtimes

Tempo runtimes provide the core functionality to deploy a tempo Model. They must provide concrete implementations for the following functionality:

| Method | Action  |
|--------|---------|
| deploy | deploy a model |
| undeploy | undeploy a model |
| wait_ready | wait for a deployed model to be ready |
| get_endpoint | get the URL for the deployed model so it can be called |
| to_k8s_yaml | optionally get the Kubernetes declarative yaml for the model |

The Runtimes defined within Tempo are:

| Runtime | Infrastructure Target | Example |
| ------- | --------------------- | -------- |
| SeldonDockerRuntime | deploy Tempo models to Docker | [intro](../examples/custom-model/local.html) |
| SeldonKubernetesRuntime | deploy Tempo models to a Kubernetes cluster with Seldon Core installed | [intro_k8s](../examples/multi-model/k8s.html) |
| KFServingKubernetesRuntime | deploy Tempo models to a Kubernetes cluster with KFServing installed | [intro](../examples/kfserving/README.html) |
| SeldonDeployRuntime | deploy Tempo models to a Kubernetes cluster with Seldon Deploy installed | |