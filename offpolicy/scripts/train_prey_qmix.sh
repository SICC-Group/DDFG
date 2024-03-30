#!/bin/sh
env="Predator_prey"
num_hare=0
num_agents=9
num_stags=6
algo="qmix"
exp="p=-1"
seed_max=127
seed_min=127
name=""

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python train/train_prey.py --user_name ${name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}  --num_agents ${num_agents} --num_hare ${num_hare} --num_stags ${num_stags} --seed ${seed} --episode_length 200 --lr 7e-4 --use_soft_update --hard_update_interval_episode 200 --tau 0.005 --num_env_steps 2000000 --miscapture_punishment -1  --eval_interval 2000 --num_eval_episodes 10 --reward_time -0.1 --use_feature_normalization --use_reward_normalization --epsilon_anneal_time 50000 --use_wandb --batch_size 8 --capture_freezes
    #--use_feature_normalization
    echo "training is done!"
done

