
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
python setup.py install
```

We need to bring `pydantic` to 1.10.8:

```
pip install pydantic==1.10.8
```

## Benchmark

We can then run

```
python examples/benchmark.py --use-dummy-weights --model=WizardLM/WizardCoder-Python-34B-V1.0 -tp 2
```
