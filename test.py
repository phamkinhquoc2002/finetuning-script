import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_process(rank, world_size):
    setup(rank, world_size)
    tensor = torch.ones(1, device=f"cuda:{rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}/{world_size}: {tensor}")
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(run_process, args=(world_size,), nprocs=world_size, join=True)