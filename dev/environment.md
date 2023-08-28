
## Environment Setup

To develop, first fork the repo to folder, say `vllm`:

```
git clone https://github.com/togethercomputer/vllm
```

Then start docker: 

```
sudo docker run --gpus all -it --rm --shm-size=24g -v ~/vllm:/workspace/vllm  nvcr.io/nvidia/pytorch:22.12-py3
```

or 

```
sudo docker run --gpus all -it --rm --shm-size=5g --network host -v ~/vllm:/workspace/vllm nvidia/cuda:12.2.0-devel-ubuntu20.04
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

We need to bring `pydantic` to 1.10.8:

```
pip install pydantic==1.10.8
```

## Benchmark

We can then run

```
RAY_DEDUP_LOGS=0 python examples/benchmark.py --use-dummy-weights --model=WizardLM/WizardCoder-Python-34B-V1.0 -tp 2
```
## Develop

It is faster to just run
```
python setup.py install
```
when there is code change
