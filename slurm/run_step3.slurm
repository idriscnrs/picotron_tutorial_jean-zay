#!/bin/bash
#SBATCH --job-name=step3 

#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=1          
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=24         
#SBATCH --hint=nomultithread 

#SBATCH --constraint=h100 
#SBATCH --account=sos@h100 

#SBATCH --qos=qos_gpu_h100-dev  
#SBATCH --time=00:10:00

#SBATCH --output=step3_dataloader_%j.out      
#SBATCH --error=step3_dataloader_%j.out      
 
# Module Time !
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0
 
set -x
export WANDB_MODE=offline

# Training Time !
srun  python ../step3_dataloader/train.py --micro_batch_size 4 --gradient_accumulation_steps 8 --seq_len 1024 --max_tokens 4096000 --num_proc 16 \
--model_name $DSDIR/HuggingFace_Models/TinyLlama/TinyLlama_v1.1 --num_hidden_layers 22 --num_attention_heads 32 --num_key_value_heads 4 \
 --run_name baseline_1B --use_wandb
