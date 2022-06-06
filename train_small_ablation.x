#!/bin/bash
#PBS -N LJW_CH_small
#PBS -l nodes=gnode7:ppn=8
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

exp_name=exp_small_ablation

source activate LJW_DeepSLIP

rm -rf ./save/${exp_name}/ 
mkdir ./save/${exp_name}/ 

python -u train_small_ablation.py \
1>./save/${exp_name}/output.txt \
2>./save/${exp_name}/error.txt

