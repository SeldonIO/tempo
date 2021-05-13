SHELL := /bin/bash 
VERSION = $(shell sed 's/^__version__ = "\(.*\)"/\1/' ./tempo/version.py)

.PHONY: install
install:
	pip install -e .

.PHONY: install-dev
install-dev: install
	pip install -r requirements-dev.txt
	pip install -r docs/requirements-docs.txt

.PHONY: test
test:
	tox

.PHONY: fmt
fmt:
	isort . \
		--skip ansible \
		--skip protoc-gen-validate \
		--skip api-common-protos \
		--skip .tox \
		--skip .eggs \
		--skip build
	black . \
		--exclude "(.eggs|.tox|ansible|build|protoc-gen-validate|api-common-protos)" \
		--line-length 120

.PHONY: lint
lint:
	flake8 . \
		--extend-exclude "ansible,protoc-gen-validate,api-common-protos,tempo/metadata"
	mypy ./tempo
	isort . --check \
		--skip ansible \
		--skip protoc-gen-validate \
		--skip api-common-protos \
		--skip tempo/metadata \
		--skip .tox \
		--skip .eggs \
		--skip build
	black . \
		--check \
		--exclude "(.eggs|.tox|ansible|build|protoc-gen-validate|api-common-protos|tempo/metadata)" \
		--line-length 120

.PHONY: install-rclone
install-rclone:
	curl https://rclone.org/install.sh | sudo bash

.PHONY: tests/testdata
tests/testdata:
	cd tests/testdata && \
		gsutil -m cp -r gs://seldon-models/sklearn . && \
		gsutil -m cp -r gs://seldon-models/xgboost . && \
		gsutil -m cp -r gs://seldon-models/mlflow . && \
		gsutil -m cp -r gs://seldon-models/keras . && \
		gsutil -m cp -r gs://seldon-models/tfserving .


.PHONY: clean_test_data
clean_test_data:
	rm -rf tests/examples


.PHONY: build
build: clean
	python setup.py sdist bdist_wheel

.PHONY: clean
clean:
	rm -rf ./dist ./build *.egg-info

.PHONY: push-test
push-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: push
push:
	twine upload dist/*

.PHONY: version
version:
	@echo ${VERSION}

protoc-gen-validate:
	git clone git@github.com:envoyproxy/protoc-gen-validate.git

api-common-protos:
	git clone git@github.com:googleapis/api-common-protos.git

build_metadata_protos: protoc-gen-validate api-common-protos
	python \
		-m grpc.tools.protoc \
		-I./ \
		-I./api-common-protos/ \
		-I./tempo/metadata/ \
		--python_out=./ \
		--grpc_python_out=./ \
		$$(find ./tempo/metadata -name '*.proto')
