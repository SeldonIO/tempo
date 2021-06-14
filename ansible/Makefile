KIND_IMAGE ?= kindest/node:v1.18.15
KIND_NAME ?= ansible

# Kind Cluster
kind-cluster:
	kind create cluster --name ${KIND_NAME} --image ${KIND_IMAGE} || echo "already there"
	kind export kubeconfig --name ${KIND_NAME}
	kubectl create ns seldon || echo "already there"
	kubectl config set-context --current --namespace seldon

kind-delete:
	kind delete cluster --name ${KIND_NAME}

# Port Forwards
port-forward-ambassador:
	kubectl port-forward -n ambassador svc/ambassador 8003:80

port-forward-istio:
	kubectl port-forward  -n istio-system svc/istio-ingressgateway 8003:80

port-forward-istio-8004:
	kubectl port-forward  -n istio-system svc/istio-ingressgateway 8004:80

port-forward-minio:
	kubectl port-forward -n minio-system svc/minio 8090:9000
