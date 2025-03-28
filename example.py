import os
import torch
import dummy_collectives
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group("cpu:gloo,cuda:gloo", rank=rank, world_size=world_size)

    # Use a different GPU for each rank
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")
    assert (rank < n_gpus), f"Rank {rank} exceeds number of GPUs {n_gpus}"
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    print(f"[Rank {rank}] Using GPU: {device}")

    # Create a tensor filled with rank value
    if rank == 0:
        value = 12345
    elif rank == 1:
        value = 54321
    tensor = torch.full((10,), value, device=device)

    # Perform all_reduce (sum)
    dist.all_reduce(tensor)
    print(f"[Rank {rank}] After all_reduce: {tensor}")

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()