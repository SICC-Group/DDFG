#!/bin/sh
env="Predator_prey"
num_hare=0
num_agents=9 #9
num_stags=6 #6
num_factor=45 #45
algo="dcg"
exp="p=-1"
name=""
seed_max=126
seed_min=126

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_prey.py --user_name ${name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --num_factor ${num_factor} --seed ${seed} --num_hare ${num_hare} --num_stags ${num_stags} --n_training_threads 2 --episode_length 200 --use_soft_update --lr 1e-3 --hard_update_interval_episode 200 --num_env_steps 2000000 --cuda  --batch_size 8 --lamda 0 --msg_iterations 4 --miscapture_punishment -1 --reward_time -0.1 --buffer_size 500 --eval_interval 2000 --num_eval_episodes 10  --epsilon_anneal_time 50000 --use_wandb --use_reward_normalization --capture_freezes 
    #--use_dyn_graph --use_reward_normalization  --prev_act_inp  --use_wandb   --use_wandb
    echo "training is done!"
done
