#!/bin/sh
env="disperse"
num_hare=0
num_agents=12
num_stags=6
algo="qplex"
exp="debug"
seed_max=188
#50,70,123 176 188 
seed_min=188
name="syc"

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train/train_prey.py --user_name ${name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}  --num_agents ${num_agents} --num_hare ${num_hare} --num_stags ${num_stags} --seed ${seed} --lr 7e-4 --use_soft_update --hard_update_interval_episode 200 --tau 0.005  --eval_interval 2000 --num_eval_episodes 10  --miscapture_punishment -1 --reward_time -0.1 --use_feature_normalization --use_reward_normalization --epsilon_anneal_time 50000 --use_wandb --capture_freezes --batch_size 8 --num_kernel 10 --adv_hypernet_layers 3 --nonlinear --use_save --catch_reward 10 --episode_length 10 --num_env_steps 1000000
    #--use_feature_normalization
    echo "training is done!"
done

