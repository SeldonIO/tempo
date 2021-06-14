# Runtimes

Tempo runtimes provide the core functionality to deploy a tempo Model. They must provide concrete implementations for the following functionality:

| Method | Action  |
|--------|---------|
| deploy | deploy a model |
| undeploy | undeploy a model |
| wait_ready | wait for a deployed model to be ready |
| endpoint | get the URL for the deployed model so it can be called |
| manifest | optionally get the Kubernetes declarative yaml for the model |

The Runtimes defined within Tempo are:

| Runtime | Infrastructure Target | Example |
| ------- | --------------------- | -------- |
| SeldonDockerRuntime | deploy Tempo models to Docker | [Custom model](../examples/custom-model/README.html) |
| SeldonKubernetesRuntime | deploy Tempo models to a Kubernetes cluster with Seldon Core installed | [Multi-model](../examples/multi-model/README.html) |
| KFServingKubernetesRuntime | deploy Tempo models to a Kubernetes cluster with KFServing installed | [KFServing](../examples/kfserving/README.html) |
| SeldonDeployRuntime | deploy Tempo models to a Kubernetes cluster with Seldon Deploy installed | |