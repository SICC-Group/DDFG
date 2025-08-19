#!/bin/sh
env="StarCraft2"
map="2s3z"
algo="mqtran"
exp="debug"
name="syc"
seed_max=33
seed_min=33

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} \
     --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
      --seed ${seed} --n_training_threads 1 --buffer_size 100000 --lr 5e-4 --batch_size 480 --use_soft_update \
       --hard_update_interval 10000 --num_env_steps 1000000 \
       --log_interval 3000 --eval_interval 10000 --user_name ${name}\
       --use_global_all_local_state --gain 1 --use_wandb
    echo "training is done!"
done
#CUDA_VISIBLE_DEVICES=2
