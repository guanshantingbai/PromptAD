#!/bin/bash
set -e

# 限制 CPU 线程数，避免多进程并行时 CPU 超载
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

nohup python run_cls.py > cls.out 2>&1 &
pid1=$!
wait $pid1

nohup python run_seg.py > seg.out 2>&1 &