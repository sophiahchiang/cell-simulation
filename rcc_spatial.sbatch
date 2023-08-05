#!/bin/zsh
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1          # Number of nodes
#SBATCH --account=pi-amurugan
#SBATCH --error=spatial_%j.err
#SBATCH --output=spatial_%j.out

# LOAD MODULES AND SPECIFY ENVIRONMENT
module load python/anaconda-2019.03
module load ffmpeg

python spatial_structure.py --net_size 170 --iter 3e5 --pyr_idx 2000 --n_reps 500 --mut_prob 0.0 0.0
python spatial_structure.py --net_size 170 --iter 3e5 --pyr_idx 2000 --n_reps 500 --mut_prob 0.0001 0.0
python spatial_structure.py --net_size 170 --iter 3e5 --pyr_idx 2000 --n_reps 500 --mut_prob 0.001 0.0
python spatial_structure.py --net_size 170 --iter 3e5 --pyr_idx 2000 --n_reps 500 --mut_prob 0.01 0.0
python spatial_structure.py --net_size 170 --iter 3e5 --pyr_idx 2000 --n_reps 500 --mut_prob 0.1 0.0

wait
exit 0

# EOF
