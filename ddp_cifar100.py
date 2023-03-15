import os
import gc
import argparse

import wandb
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100
from tqdm.autonotebook import tqdm
from time import perf_counter

from syncbn import SyncBatchNorm


torch.set_num_threads(1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

def init_process(local_rank, fn, backend="nccl", **kwargs):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size, **kwargs)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        hidden_size = 1024
        self.fc1 = nn.Linear(6272, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 100)
        # self.bn1 = nn.SyncBatchNorm(hidden_size, affine=False)
        self.bn1 = SyncBatchNorm(hidden_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model, accumulation_steps=1):
    size = float(dist.get_world_size()) * accumulation_steps
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run_training(rank, size, grad_accumulation_steps, device, batch_size, num_workers):
    device += f':{rank}'
    torch.manual_seed(0)

    train_dataset = CIFAR100(
        "./cifar", transform=transform, download=True, train=True,
    )
    val_dataset = CIFAR100(
        "./cifar", transform=transform, download=True, train=False,
    )
    # where's the validation dataset?
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset, size, rank),
                              batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset, size, rank),
                            batch_size=batch_size, num_workers=num_workers)

    model = Net()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    batch_number = 0

    for epoch in range(10):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
        start_time = perf_counter()
        epoch_loss = torch.zeros(1, device=device)

        for data, target in tqdm(train_loader, total=len(train_loader), desc=f'Train {epoch}'):
            batch_number += 1
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.detach()
            loss.backward()

            if batch_number % grad_accumulation_steps == 0:
                average_gradients(model, grad_accumulation_steps)
                optimizer.step()
                optimizer.zero_grad()

            acc = (output.argmax(dim=1) == target).float().mean()

            # print(f"Rank {dist.get_rank()}, loss: {epoch_loss / num_batches}, acc: {acc}")

        torch.cuda.synchronize()
        elapsed_time = perf_counter() - start_time

        val_loss = torch.zeros(1, device=device)
        hits = torch.zeros(1, device=device)
        samples = torch.zeros(1, device=device)
        for data, target in tqdm(val_loader, total=len(val_loader), desc=f'Validation {epoch}'):
            data = data.to(device)
            target = target.to(device)

            with torch.no_grad():
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target, reduction='sum')
                val_loss += loss
                hits += (output.argmax(dim=1) == target).float().sum()
                samples += output.size(0)

        msg_tensor = torch.cat([val_loss, hits, samples], dim=0)
        dist.all_reduce(msg_tensor,  dist.ReduceOp.SUM)
        global_val_loss, global_hits, global_samples = torch.split(msg_tensor, 1)

        if local_rank == 0:
            memory = torch.cuda.memory_allocated() / 1024 / 1024
            wandb.log(dict(
                    train_loss=epoch_loss/len(train_loader), 
                    global_val_loss=global_val_loss / global_samples,
                    global_accuracy=global_hits / global_samples,
                    elapsed_time=elapsed_time, 
                    memory=memory,
            ))

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--grad-accumulation-steps', default=1, type=int)
    argparser.add_argument('--batch-size', type=int, default=64)
    argparser.add_argument('--num-workers', type=int, default=0)
    argparser.add_argument('--device', type=str, default='cpu')
    argparser.add_argument('--name', type=str, default='DEBUG')
    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == 0:
        run = wandb.init(
            entity='i_vainn',
            project="parallel-training",
            name=args.name,
            config=args.__dict__
        )
    
    init_process(
        local_rank=local_rank, fn=run_training,
        backend="gloo" if args.device == 'cpu' else 'nccl',
        grad_accumulation_steps=args.grad_accumulation_steps, 
        device=args.device, batch_size=args.batch_size, num_workers=args.num_workers
    )
