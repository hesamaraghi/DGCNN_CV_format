#!/bin/bash
# TODO: put your desired HPC preamble: partition, qos, time, gres, output, etc.
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=END


# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi


echo "Input to sbatch: " $@

module use /opt/insy/modulefiles
module load cuda/11.4 cudnn/11.4-8.2.2.26

# TODO: activate your own conda environment here

# TODO: insert your W&B API key
export WANDB_API_KEY="81ddbde7aa344dda1cd4056e697173f76038c394"

# For multi-GPU, don't use P2P as it hangs
export NCCL_P2P_DISABLE=1

srun "$@"

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

