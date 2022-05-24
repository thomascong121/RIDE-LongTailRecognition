#!/bin/bash
#PBS -P sz65
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -l mem=64GB
#PBS -l storage=scratch/sz65
#PBS -l walltime=24:00:00
#PBS -l jobfs=100GB

args=$(echo ${args}| envsubst)
args=${args//|/ }
echo $args

module load pytorch/1.9.0
cd /scratch/sz65/cc0395/LT_ECCV
source my_env/bin/activate
cd /scratch/sz65/cc0395/RIDE

echo "Script executed from: ${PWD}"

export WANDB_DIR=/scratch/sz65/cc0395/LT_ECCV
export WANDB_CACHE_DIR=/scratch/sz65/cc0395/wandb/.cache
export WANDB_CONFIG_DIR=/scratch/sz65/cc0395/wandb/.config
export WANDB_MODE=offline
python3 train.py dataset.imb_factor=${IMFCT_decm} run.normal.batch_size=128 run.ddp.dist_url="tcp://localhost:$PORT" run.normal.epochs=400 run.normal.warmup_epochs=10 $args

python train.py -c "configs/config_imbalance10_cifar10_ride.json" --reduce_dimension 1 --num_experts 3 --epochs 5
python train.py -c "configs/config_imbalance_cifar10_ride_ea.json" -r /scratch/sz65/cc0395/RIDE/saved/models/Imbalance10_CIFAR10_LT_RIDE/0524_153259/model_best.pth --reduce_dimension 1 --num_experts 3
python test.py -r /scratch/sz65/cc0395/RIDE/saved/models/Imbalance_CIFAR10_LT_RIDE_EA/0524_153651/model_best.pth


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