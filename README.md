## conda environment

```
conda create -n zezhou-pytorch-test python=3.12.9
conda activate zezhou-pytorch-test
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install numpy
```

Then running `python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"` should print:

```
2.5.1+rocm6.2
True
```

## Build

```python setup.py develop```

## Usage

```python
import os

import torch
import dummy_collectives

import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group("cpu:gloo,cuda:dummy", rank=0, world_size=1)

# this goes through gloo
x = torch.ones(6)
dist.all_reduce(x)
print(f"cpu allreduce: {x}")

# this goes through dummy
if torch.cuda.is_available():
    y = x.cuda()
    dist.all_reduce(y)
    print(f"cuda allreduce: {y}")

    try:
        dist.broadcast(y, 0)
    except RuntimeError:
        print("got RuntimeError when calling broadcast")
```