#!/bin/bash
#
#SBATCH --job-name=fl-3round-simulation # Job name for tracking
#SBATCH --partition=eagle     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6     # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=4000rounds_multirun_FL_%j.out
#SBATCH --error=2000rounds_multirun_FL_%j.err


## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export RAY_DEDUP_LOGS=0
## Execute your program(s) ##

code_path="/dcs/large/u2238887/Dissertation/flwr_simulations/adaptive_federated_optimization"


conda activate flwr
module load CUDA

# Start an interactive session
#srun --pty bash


#srun python ${code_path}/scripts/main.py --config-path ${code_path}/conf/cifar10 strategy=fedavg num_rounds=3 num_total_clients=100 cpus_per_client=2 gpus_per_client=0.0001
srun python ${code_path}/scripts/main.py -m --config-path ${code_path}/conf/cifar10 strategy=fedyogi,fedadagrad,fedadam,fedavg num_rounds=4000 num_total_clients=100 cpus_per_client=2 gpus_per_client=0.0001

#srun python3.9 my_cuda_program.py
