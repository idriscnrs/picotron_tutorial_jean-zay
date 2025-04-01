# Picotron tutorial Jean Zay !

A step by step tutorial on how to build [Picotron](https://github.com/huggingface/picotron) distributed training framework form scratch 🔥

## Videos

> More to come. Full playlist [here](https://www.youtube.com/playlist?list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S) 🎬

- 🎬 [[Picotron tutorial] Part 1: Model, Process Group Manager, Dataloader](https://youtu.be/u2VSwDDpaBM)
- 🎬 [[Picotron tutorial] Part 2: Tensor Parallel](https://www.youtube.com/watch?v=qUMPaSWi5HI&list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S&index=3)
- 🎬 [[Picotron tutorial] Bonus: Debugging Distributed codebase](https://www.youtube.com/watch?v=_8xlRgFY_-g&list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S&index=4)
- 🎬 [[Picotron tutorial] Part 3: Data Parallel (Naive & Bucket)](https://www.youtube.com/watch?v=k8EpWveM_t4&list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S&index=4)


## Jean Zay fork 
*This fork was build to run on H100 Jean Zay's partition. It could work on A100 with some adjustment*

### Setup 
The support team make things easy, so we got you a great software environnement.

```bash
module load arch/h100
module load pytorch-gpu/py3/2.5.0
```

### idr_torch
This fork inclute `idr_torch` library to get slurm variable easily.

### Dataset & Model config
You can't access internet from the compute node.
We fetch dataset & model direcly from our $DSDIR storage (local).

### Wandb
You can't access internet from the compute node.
We disable wandb connectivity.
```bash
export WANDB_MODE=offline
```

### Job submissions
Jean Zay is using slurm to schedule jobs on the cluster.
You can find slurm submission scripts in `slurm/` folder.
The provided script are for H100 partition, you will need an account @h100 to submit your job.

```bash
# Basline
sbatch run_step3.slurm

# Tensor Parallel
sbatch run_step4.slurm

# Data Parallel
sbatch run_step6.slurm

# Pipeline Parallel
sbatch run_step8.slurm

# 3D parallelism (Tensor + Data + Pipeline parallel)
sbatch run_step9.slurm
```
