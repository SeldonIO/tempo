SHELL := /bin/bash 
VERSION = $(shell sed 's/^__version__ = "\(.*\)"/\1/' ./tempo/version.py)

.PHONY: install
install:
	pip install -e .

.PHONY: install-dev
install-dev: install
	pip install -r requirements-dev.txt
	pip install -r doc/requirements-docs.txt

.PHONY: test
test:
	tox

.PHONY: fmt
fmt:
	isort . \
		--skip ansible \
		--skip .tox \
		--skip .eggs \
		--skip build
	black . \
		--exclude "(.eggs|.tox|ansible|build)" \
		--line-length 120

.PHONY: lint
lint:
	flake8 . \
		--extend-exclude "ansible"
	mypy ./tempo
	isort . --check \
		--skip ansible \
		--skip .tox \
		--skip .eggs \
		--skip build
	black . \
		--check \
		--exclude "(.eggs|.tox|ansible|build)" \
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
