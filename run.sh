#!/bin/bash
set -e

for target in env soc gov
do
  echo "=== Training target: $target ==="
  python train.py \
    --target $target \
    --epochs 100 \
    --batch_size 1 \
    --lr 1e-4 \
    --alpha 0.25 \
    --lambd 10.0 \
    --device cuda \
    --logdir runs
done

echo "done. Best checkpoints:"
ls -lh best_dynscan_*.pt
   