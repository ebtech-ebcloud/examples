#!/bin/bash


#export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # 启用IB
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107  # 指定IB设备
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0  # Bootstrap网络接口
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

export NCCL_PORT_RANGE=40000-50000
export NCCL_IB_PORT_RANGE=40000-50000
export NCCL_P2P_PORT_RANGE=40000-50000

# 设置主节点地址（node0的IP）
MASTER_ADDR="10.233.127.255"
MASTER_PORT="12355"
WORLD_SIZE=16

# 在当前机器上启动8个进程（rank 0-7）
for ((i=0; i<8; i++)); do
    RANK=$i
    echo "Launching rank $RANK"
    python nccl_test_2_node.py \
        --master-addr $MASTER_ADDR \
        --master-port $MASTER_PORT \
        --world-size $WORLD_SIZE \
        --rank $RANK &
done

wait
