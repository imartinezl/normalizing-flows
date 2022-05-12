#!/bin/bash
#SBATCH -J main              # Jobname
#SBATCH --nodes=1            # Nodes requested
#SBATCH --ntasks=1           # Tasks requested
#SBATCH --time=1-00:00:00   # Walltime hour:minute:second
#SBATCH --mem=12GB
#SBATCH --partition=interactive  # Queue
#SBATCH --gres=gpu:1

salloc --ntasks=1 --partition=interactive --time=60  --mem=12GB --gpus=1 --nodelist=abacus001 --cpus-per-task=8

module load Miniconda3/4.9.2
module load CUDA/10.2.89-GCC-8.3.0 # for CPAB
source activate .venv

python nf_1D_args.py  --folder results1D --dataset UNIFORM --train-size 5000 --test-size 2000  --batch-size 256 --tess-size 4 --flow-steps 1 --epochs 500 --lr 0.0001 

python nf_2D_args.py  --folder results2D --dataset MOONS --train-size 5000 --test-size 2000  --batch-size 256 --hidden-dim 5 --hidden-layers 1 --tess-size 4 --flow-steps 1 --epochs 500 --lr 0.0005 

python nf_ND_args.py  --folder resultsND --dataset SCURVE --train-size 15000 --test-size 5000  --batch-size 256 --hidden-dim 8 --hidden-layers 1 --tess-size 4 --flow-steps 1 --epochs 1 --lr 0.001 

srun --pty --jobid $JOBID -w $NODEID /bin/bash
srun --pty --jobid $JOBID bash
nvidia-smi -l 1
watch -n1 nvidia-smi

# pip install -r requirements.txt
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser


T = cpab.Cpab(10, backend="pytorch", device="gpu")
T.uniform_meshgrid(100)