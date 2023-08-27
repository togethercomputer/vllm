
## Environment Setup

To develop, first fork the repo to folder, say `vllm`:

```
git clone https://github.com/togethercomputer/vllm
```

Then start docker: 

```
sudo docker run --gpus all -it --rm --shm-size=24g -v ~/vllm:/workspace/vllm  nvcr.io/nvidia/pytorch:22.12-py3
```

Upgrade vllm to latest version
```
pip install --upgrade pip
```

Then install
```
cd vllm
pip install -e .
```
