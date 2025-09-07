import torch
import torch.distributed as dist
import os
import argparse
import time
import numpy as np

import os

# 打印所有环境变量
for key, value in os.environ.items():
    print(f"{key}: {value}")

print("----------------------------")

def setup(rank, world_size, master_addr, master_port):
    """初始化进程组"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # 使用NCCL后端
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())

def cleanup():
    """清理进程组"""
    dist.destroy_process_group()

def test_all_reduce(rank, world_size, master_addr, master_port, tensor_size):
    """测试All-Reduce操作"""
    setup(rank, world_size, master_addr, master_port)
    
    # 创建GPU张量 - 使用浮点数确保精度
    tensor = torch.ones(tensor_size, dtype=torch.float32, device=f'cuda:{rank % torch.cuda.device_count()}') * (rank + 1.0)
    original_value = tensor[0].item()
    
    # 同步所有进程
    dist.barrier()
    
    if rank == 0:
        print(f"Testing All-Reduce with tensor size: {tensor_size}")
        print(f"Before all-reduce: Rank {rank} has first element: {original_value}")
    
    # 计时开始
    torch.cuda.synchronize()
    start_time = time.time()
    
    # 执行All-Reduce操作
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # 计时结束
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算正确的期望值
    expected_value = sum(range(1, world_size + 1))  # 1+2+3+...+16 = 136
    actual_value = tensor[0].item()
    
    if rank == 0:
        print(f"After all-reduce: Rank {rank} has first element: {actual_value}")
        print(f"Expected value: {expected_value}")
        print(f"All-Reduce time: {(end_time - start_time) * 1000:.3f} ms")
        
        # 正确计算带宽：数据量（字节）* 2（发送+接收） / 时间
        data_size_bytes = tensor.element_size() * tensor.nelement()
        bandwidth = (data_size_bytes * 2) / (end_time - start_time) / 1e9  # GB/s
        print(f"Bandwidth: {bandwidth:.3f} GB/s")
        
        # 使用容差比较浮点数
        if abs(actual_value - expected_value) < 1e-6:
            print("All-Reduce Test PASSED!")
        else:
            print(f"All-Reduce Test FAILED! Difference: {abs(actual_value - expected_value)}")
    
    cleanup()

def test_broadcast(rank, world_size, master_addr, master_port, tensor_size):
    """测试Broadcast操作"""
    setup(rank, world_size, master_addr, master_port)
    
    # 创建GPU张量
    if rank == 0:  # 只有rank 0初始化数据
        tensor = torch.ones(tensor_size, dtype=torch.float32, device=f'cuda:{rank % torch.cuda.device_count()}') * 42.0
    else:
        tensor = torch.zeros(tensor_size, dtype=torch.float32, device=f'cuda:{rank % torch.cuda.device_count()}')
    
    original_value = tensor[0].item()
    
    # 同步所有进程
    dist.barrier()
    
    if rank == 0:
        print(f"\nTesting Broadcast with tensor size: {tensor_size}")
        print(f"Before broadcast: Rank {rank} has data: {original_value}")
    
    # 计时开始
    torch.cuda.synchronize()
    start_time = time.time()
    
    # 执行Broadcast操作
    dist.broadcast(tensor, src=0)
    
    # 计时结束
    torch.cuda.synchronize()
    end_time = time.time()
    
    final_value = tensor[0].item()
    
    if rank == 0:
        print(f"Broadcast time: {(end_time - start_time) * 1000:.3f} ms")
        
        # 正确计算带宽：数据量（字节） / 时间
        data_size_bytes = tensor.element_size() * tensor.nelement()
        bandwidth = data_size_bytes / (end_time - start_time) / 1e9  # GB/s
        print(f"Bandwidth: {bandwidth:.3f} GB/s")
    
    # 所有rank都检查结果
    if abs(final_value - 42.0) < 1e-6:
        if rank == 0:
            print("Broadcast Test PASSED!")
    else:
        print(f"Rank {rank} Broadcast Test FAILED! Got: {final_value}, Expected: 42.0")
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='NCCL Communication Test')
    parser.add_argument('--master-addr', type=str, required=True, help='Master node address')
    parser.add_argument('--master-port', type=str, default='12355', help='Master node port')
    parser.add_argument('--world-size', type=int, default=16, help='Total number of processes')
    parser.add_argument('--rank', type=int, required=True, help='Rank of the current process')
    parser.add_argument('--tensor-size', type=int, default=1000000, help='Size of tensor for testing')
    
    args = parser.parse_args()
    
    print(f"Rank {args.rank} starting on device {args.rank % torch.cuda.device_count()}")
    
    # 测试All-Reduce
    test_all_reduce(args.rank, args.world_size, args.master_addr, args.master_port, args.tensor_size)
    
    # 测试Broadcast
    test_broadcast(args.rank, args.world_size, args.master_addr, args.master_port, args.tensor_size)

if __name__ == "__main__":
    main()
