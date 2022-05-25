#!/bin/bash
#PBS -P sz65
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -l mem=64GB
#PBS -l storage=scratch/sz65
#PBS -l walltime=24:00:00
#PBS -l jobfs=100GB

module load pytorch/1.9.0
cd /scratch/sz65/cc0395/LT_ECCV
source my_env/bin/activate
cd /scratch/sz65/cc0395/RIDE

echo "Script executed from: ${PWD}"

export WANDB_DIR=/scratch/sz65/cc0395/RIDE
export WANDB_CACHE_DIR=/scratch/sz65/cc0395/wandb/.cache
export WANDB_CONFIG_DIR=/scratch/sz65/cc0395/wandb/.config
export WANDB_MODE=offline

python train.py -c "configs/config_imbalance200_cifar10_ride.json" --reduce_dimension 1 --num_experts 3
python train.py -c "configs/config_imbalance200_cifar10_distill_ride.json" --reduce_dimension 1 --num_experts 3 --distill_checkpoint /scratch/sz65/cc0395/RIDE/saved/models/Imbalance200_CIFAR10_LT_RIDE/model_best.pth
python train.py -c "configs/config_imbalance200_cifar10_ride_ea.json" -r /scratch/sz65/cc0395/RIDE/saved/models/Imbalance200_CIFAR10_LT_RIDE_DISTILL/model_best.pth --reduce_dimension 1 --num_experts 3
python test.py -r /scratch/sz65/cc0395/RIDE/saved/models/Imbalance200_CIFAR10_LT_RIDE_EA/model_best.pth

#python train.py -c "configs/config_imbalance200_cifar10_ride.json" --reduce_dimension 1 --num_experts 3
#python train.py -c "configs/config_imbalance200_cifar10_ride_ea.json" -r /scratch/sz65/cc0395/RIDE/saved/models/Imbalance200_CIFAR10_LT_RIDE/model_best.pth --reduce_dimension 1 --num_experts 3
#python test.py -r /scratch/sz65/cc0395/RIDE/saved/models/Imbalance200_CIFAR10_LT_RIDE_EA/model_best.pth


# qsub -I -qgpuvolta  -Psz65 -lwalltime=01:00:00,ncpus=12,ngpus=1,mem=64GB,jobfs=1GB,wd
#python3 train.py \
#  --arch ViT-B_16 \
#  --imb-factor ${IMFCT_decm}\
#  --wd 5e-4 \
#  --class_max 5000 \
#  --alpha 100 \
#  --beta 1.0 \
#  --mark ${mask_feature}\
#  --batch_size 128\
#  --feat_dim 64 \
#  --dist-url "tcp://localhost:$PORT" \
#  --epochs 10\
#  --lr_strategy 'cosine'\
#  --warmup_epochs 5 $args