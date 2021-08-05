FROM docker.io/seldonio/seldon-core-s2i-python37-ubi8:1.5.0
LABEL name="Seldon MLOPs Utils" \
      vendor="Seldon Technologies" \
      version="0.1" \
      release="1" \
      summary="Seldon MLOPs Utils" \
      description="Artifact handling utilities"

RUN pip install pip -U

COPY mlops mlops
COPY setup.py .
COPY README.md README.md

RUN pip install -e .

COPY mlops.py mlops.py

ENTRYPOINT ["python", "mlops.py"]

