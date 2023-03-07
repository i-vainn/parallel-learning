import os
import random

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from time import perf_counter


def init_process(rank, size, fn, master_port, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def butterfly_allreduce(send, rank, size):
    """
    Performs Butterfly All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """

    buffer_for_chunk = torch.empty_like(send)

    send_futures = []

    for i, elem in enumerate(send):
        if i != rank:
            send_futures.append(dist.isend(elem, i))

    recv_futures = []

    for i, elem in enumerate(buffer_for_chunk):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))
        else:
            elem.copy_(send[i])

    for future in recv_futures:
        future.wait()

    # compute the average
    torch.mean(buffer_for_chunk, dim=0, out=send[rank])

    for i in range(size):
        if i != rank:
            send_futures.append(dist.isend(send[rank], i))

    recv_futures = []

    for i, elem in enumerate(send):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))

    for future in recv_futures:
        future.wait()
    for future in send_futures:
        future.wait()


def ring_allreduce(send, rank, size):
    """
    Performs Ring All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """
    buffer_for_chunk = torch.empty_like(send)
    
    sent = []
    total_steps = 2 * (size - 1)
    send_rk = (rank + 1) % size
    recv_rk = (rank - 1) % size

    for step in range(1, total_steps + 1):    
        send_idx = (rank - step + 1) % size
        recv_idx = (rank - step) % size
        sent.append(dist.isend(send[send_idx], send_rk))
        req = dist.irecv(buffer_for_chunk[recv_idx], recv_rk)
        req.wait()

        if step <= total_steps // 2:
            send[recv_idx] += buffer_for_chunk[recv_idx]
        else:
            send[recv_idx] = buffer_for_chunk[recv_idx]
    
    for req in sent:
        req.wait()
    
    send /= size
    

def run_butterfly_allreduce(rank, size):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((size,), dtype=torch.float)
    print("Rank ", rank, " has data ", tensor)
    butterfly_allreduce(tensor, rank, size)
    print("Rank ", rank, " has data ", tensor)


def run_ring_allreduce(rank, size):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((size,), dtype=torch.float)
    print("Rank ", rank, " has data ", tensor)
    ring_allreduce(tensor, rank, size)
    print("Rank ", rank, " has data ", tensor)


def run_any_allreduce(allreduce, rank, world_size, tensor_size):
    torch.manual_seed(rank)
    tensor = torch.randn((world_size, tensor_size // world_size), dtype=torch.float)
    allreduce(tensor, rank, world_size)


def init_measurement(rank, world_size, tensor_size, fn, master_port, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    run_any_allreduce(fn, rank, world_size, tensor_size)


def measure_communication_time(allreduce, world_size, tensor_size):
    start_time = perf_counter()

    processes = []
    port = random.randint(25000, 30000)
    for rank in range(world_size):
        p = Process(
            target=init_measurement,
            args=(rank, world_size, tensor_size, allreduce, port)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    end_time = perf_counter() - start_time
    return end_time


if __name__ == "__main__":
    size = 4
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_butterfly_allreduce, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
