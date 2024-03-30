#!/bin/sh
env="StarCraft2"
map="5m_vs_6m"
algo="wqmix"
exp="ow-qmix"
name="syc"
seed_max=56
seed_min=56

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train/train_smac.py --env_name ${env} \
     --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
      --seed ${seed} --n_training_threads 1 --buffer_size 5000 --lr 1e-3 --batch_size 8 --use_soft_update \
       --hard_update_interval_episode 200 --num_env_steps 5000000 \
       --log_interval 3000 --eval_interval 20000 --user_name ${name}\
       --use_global_all_local_state --gain 1 --use_wandb --use_save --epsilon_anneal_time 50000 --hysteretic_qmix
    echo "training is done!"
done
# --hysteretic_qmix   --epsilon_anneal_time 1000000
