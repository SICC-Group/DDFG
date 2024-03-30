#!/bin/sh
env="StarCraft2"
map="6h_vs_8z"
algo="qplex"
exp="debug"
name="syc"
seed_max=215
seed_min=215

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train/train_smac.py --env_name ${env} \
     --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
      --seed ${seed} --n_training_threads 1 --buffer_size 5000 --lr 1e-4 --batch_size 8 --use_soft_update \
       --hard_update_interval_episode 200 --num_env_steps 5000000 \
       --log_interval 3000 --eval_interval 20000 --user_name ${name}\
       --use_global_all_local_state --gain 1 --use_wandb --num_kernel 4 --adv_hypernet_layers 1 --nonlinear --use_save
    echo "training is done!"
done
#CUDA_VISIBLE_DEVICES=2
