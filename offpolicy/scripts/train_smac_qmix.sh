#!/bin/sh
env="StarCraft2"
map="8m_vs_9m"
algo="qmix"
exp="debug"
name=""
seed_max=126
seed_min=126

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train/train_smac.py --env_name ${env} \
     --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
      --seed ${seed} --n_training_threads 1 --buffer_size 5000 --lr 1e-3 --batch_size 8 --use_soft_update \
       --hard_update_interval_episode 200 --num_env_steps 2000000 \
       --log_interval 3000 --eval_interval 20000 --user_name ${name}\
       --use_global_all_local_state --gain 1 --use_wandb
    echo "training is done!"
done

