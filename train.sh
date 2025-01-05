#!/bin/bash

#SBATCH --job-name color_style_transfer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -o logs/slurm-%A-%x.out

python train.py --epoch=40 --content_dir=/data/lhayoung9/local_datasets/train2017/train2017 --log_dir=/data/lhayoung9/repos/color_style_transfer/logs

# letting slurm know this code finished without any problem
exit 0