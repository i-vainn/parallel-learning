import os

import torch.distributed as dist
import torch

def run_sequential(rank, size, num_iter=10):
    """
    Prints the process rank sequentially according to its number over `num_iter` iterations,
    separating the output for each iteration by `---`
    Example (3 processes, num_iter=2):
    ```
    Process 0
    Process 1
    Process 2
    ---
    Process 0
    Process 1
    Process 2
    ```
    """
    for it in range(num_iter):
        for cur_rank in range(size):
            if rank == cur_rank:
                [dist.barrier() for _ in range(rank)]
                print(f'Process {rank}', flush=True)
                [dist.barrier() for _ in range(rank, size)]
        
        if rank + 1 == size and it + 1 < num_iter:
            print('---', flush=True)
            dist.barrier()
        else:
            dist.barrier()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(rank=local_rank, backend="gloo")

    run_sequential(local_rank, dist.get_world_size())