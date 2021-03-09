SHELL := /bin/bash 
VERSION = $(shell sed 's/^__version__ = "\(.*\)"/\1/' ./tempo/version.py)

.PHONY: install
install:
	pip install -e .

.PHONY: install-dev
install-dev:
	pip install -r requirements-dev.txt

.PHONY: test
test:
	tox

.PHONY: fmt
fmt:
	black ./ --exclude "(.eggs|.tox)"


.PHONY: lint
lint:
	flake8 .
	mypy ./tempo

.PHONY: install-rclone
install-rclone:
	curl https://rclone.org/install.sh | sudo bash

.PHONY: tempo/tests/examples
tempo/tests/examples:
	mkdir -p tempo/tests/examples
	cd tempo/tests/examples && \
		gsutil cp -r gs://seldon-models/sklearn . && \
		gsutil cp -r gs://seldon-models/xgboost . && \
		gsutil cp -r gs://seldon-models/mlflow . && \
		gsutil cp -r gs://seldon-models/keras . && \
		gsutil cp -r gs://seldon-models/tfserving .


.PHONY: clean_test_data
clean_test_data:
	rm -rf tempo/tests/examples


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
