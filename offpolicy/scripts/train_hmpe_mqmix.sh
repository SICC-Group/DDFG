#!/bin/sh
env="hmpe"
algo="mqmix"
exp="debug"
name="syc"
seed_max=415
seed_min=415

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_hmpe.py --env_name ${env} \
     --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
      --seed ${seed} --n_training_threads 1 --buffer_size 100000 --lr 5e-4 --batch_size 2048 --use_soft_update \
       --hard_update_interval 20000 --num_env_steps 1000000 \
       --log_interval 6000 --eval_interval 10000 --user_name ${name}\
       --use_global_all_local_state --gain 1 --use_wandb --train_interval 300 --use_reward_normalization
    echo "training is done!"
done

