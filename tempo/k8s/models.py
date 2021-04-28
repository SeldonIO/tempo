from tempo.serve.metadata import ModelListing


class KubernetesModelListing(ModelListing):

    namespace: str
