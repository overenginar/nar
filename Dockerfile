FROM continuumio/anaconda3:latest

RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates kmod \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /narlab

COPY . .

COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

RUN pip install mlflow==2.3.1

EXPOSE 8888/tcp 5000/tcp 5001/tcp 5002/tcp

ENTRYPOINT jupyter-lab --notebook-dir=/narlab --ip=* --port=8888 --no-browser --allow-root
