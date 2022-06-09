#!/bin/bash
#SBATCH -J mainND
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=14-00:00:00
#SBATCH --partition=gpu
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --nodelist=abacus001
#SBATCH --cpus-per-task=8

# conda-env remove .venv
srun echo "############# PYTHON VIRTUAL ENVIRONMENT #############"


module load Miniconda3/4.9.2
# if ! (conda env list | grep ".venv") ; then 
#     conda create --name .venv --clone base python=3.7
# fi
module load CUDA/10.2.89-GCC-8.3.0 # for CPAB
source activate .venv

# pip install -r requirements.txt

srun echo "############# RUN SCRIPT #############"

# srun python nf_ND_args.py  --folder resultsND --dataset HEPMASS --train-size 30000 --test-size 3000  --batch-size 512 --hidden-dim 64 --hidden-layers 4 --tess-size 10 --flow-steps 5  --epochs 256 --lr 0.0005 --model-type CL 
# srun python nf_ND_args.py  --folder resultsND --dataset HEPMASS --train-size 30000 --test-size 3000  --batch-size 512 --hidden-dim 64 --hidden-layers 4 --tess-size 10 --flow-steps 5  --epochs 256 --lr 0.0005 --model-type AR
# srun python nf_ND_args.py  --folder resultsND --dataset HEPMASS --train-size 30000 --test-size 3000  --batch-size 512 --hidden-dim 64 --hidden-layers 4 --tess-size 10 --flow-steps 10 --epochs 256 --lr 0.0005 --model-type CL 
# srun python nf_ND_args.py  --folder resultsND --dataset HEPMASS --train-size 30000 --test-size 3000  --batch-size 512 --hidden-dim 64 --hidden-layers 4 --tess-size 10 --flow-steps 10 --epochs 256 --lr 0.0005 --model-type AR
# srun python nf_ND_args.py  --folder resultsND --dataset HEPMASS --train-size 30000 --test-size 3000  --batch-size 512 --hidden-dim 64 --hidden-layers 4 --tess-size 10 --flow-steps 15 --epochs 256 --lr 0.0005 --model-type CL 
# srun python nf_ND_args.py  --folder resultsND --dataset HEPMASS --train-size 30000 --test-size 3000  --batch-size 512 --hidden-dim 64 --hidden-layers 4 --tess-size 10 --flow-steps 15 --epochs 256 --lr 0.0005 --model-type AR


srun python nf_ND_args.py  --folder resultsND --dataset HEPMASS --train-size 30000 --test-size 5000  --batch-size 512 --hidden-dim 64 --hidden-layers 4 --tess-size 20 --flow-steps 10 --epochs 325 --lr 0.0005 --theta-max 1.5 --model-type CL 
srun python nf_ND_args.py  --folder resultsND --dataset HEPMASS --train-size 30000 --test-size 5000  --batch-size 512 --hidden-dim 64 --hidden-layers 4 --tess-size 20 --flow-steps 10 --epochs 325 --lr 0.0005 --theta-max 1.5 --model-type AR 
