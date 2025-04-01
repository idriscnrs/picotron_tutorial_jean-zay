#!/bin/bash
#SBATCH --job-name=step1 

#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=1          
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=24         
#SBATCH --hint=nomultithread 

#SBATCH --constraint=h100 
##SBATCH --account=XXX@h100 

#SBATCH --qos=qos_gpu_h100-dev  
#SBATCH --time=00:10:00

#SBATCH --output=step1_modeling_%j.out      
#SBATCH --error=step1_modeling_%j.out       
 
# Module Time !
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0
 
set -x
 
# Training Time !
srun  python ../step1_modeling/train.py
