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

srun python setup_experiments_ND.py
srun bash runND.sh
srun python nf_ND_results.py
