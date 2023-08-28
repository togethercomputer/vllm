
import time
import multiprocessing

import cupy
from cupy import cuda
from cupy.cuda import nccl
from cupy import testing

REPEAT = 1

def f(n_devices, device, comm_id, rank):

    device.use()

    comm = nccl.NcclCommunicator(n_devices, comm_id, rank)

    sizes = [65536,]
    for i in range(0, 15):
        sizes.append(65536 * (2**i))

    for ts in sizes:
        sendbuf = cupy.ones(ts, dtype='float16')

        for i in range(0,10):
            start = time.time()
            for repeat in range(0, REPEAT):
                comm.allReduce(sendbuf.data.ptr, sendbuf.data.ptr, sendbuf.size, 
                    nccl.NCCL_HALF, 0, cuda.Stream.null.ptr)
            device.synchronize()
            end = time.time()

            if rank == 0:
                print(ts, (end - start) / REPEAT)


if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')

    n_devices = 4
    devices = [cuda.Device(i) for i in range(n_devices)]

    comm_id = nccl.get_unique_id()

    ps = []
    for i in range(0, n_devices):
        p = multiprocessing.Process(
            target=f, args=(n_devices, devices[i], comm_id, i))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()

    print('Rank 0 successfully finished.')


# mpiexec -n N python this_script.py

"""
import cupy
from cupy import cuda
from cupy.cuda import nccl
from cupy import testing
from mpi4py import MPI


commm = MPI.COMM_WORLD
rank = commm.Get_rank()
size = commm.Get_size()

n_devices = commm.Get_size()
devices = [cuda.Device(i) for i in range(n_devices)]
comm_id = nccl.get_unique_id()

comm = nccl.NcclCommunicator(n_devices, comm_id, rank)
print(comm)

def run(n_devices, device, comm_id, rank):

    print("1")
    device.use()
    print("2")
    
    e = cupy.ones(10, dtype='float32')
    if rank == 0:
        print(e)

    comm.all_reduce(e.data.prt, e.data.prt, x.size, nccl.NCCL_FLOAT, 0, cuda.Stream.null.ptr)

    if rank == 0:
        print(e)

    device.synchronize()

run(n_devices, devices[rank], comm_id, rank)
"""

'''
import time
import cupy
from mpi4py import MPI

comm = MPI.COMM_WORLD


REPEAT = 10

sizes = [65536,]
for i in range(0, 14):
    sizes.append(65536 * (2**i))

for ts in sizes:
    sendbuf = cupy.ones(ts)

    start = time.time()
    for repeat in range(0, REPEAT):
        comm.Allreduce(sendbuf, sendbuf)
    end = time.time()

    if rank == 0:
        print(ts, (end - start) / REPEAT)
'''

"""

#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time


REPEAT = 10

def run(rank, size):

    group = dist.new_group([0, 1, 2, 3, 4, 5, 6, 7])

    device = torch.device(f"cuda:{rank}")
  
    sizes = []
    for i in range(0, 14):
        sizes.append(65536 * (2**i))

    for ts in sizes:
        tensor = torch.ones(ts, dtype=torch.float16).to(device)
        tensor = tensor / 1024

        start = time.time()
        for repeat in range(0, REPEAT):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        end = time.time()

        print(tensor)

        if rank == 0:
            print(ts, (end - start) / REPEAT)

    pass

def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 8
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
"""