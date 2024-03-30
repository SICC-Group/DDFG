#!/bin/sh
env="Predator_prey"
num_hare=0
num_agents=9  #9
num_stags=6 #6
num_factor=18  #18
algo="rddfg_cent_rw"
exp="debug"
name="syc"
seed_max=703
seed_min=703

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_prey.py --user_name ${name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --num_agents ${num_agents} --num_factor ${num_factor} --seed ${seed} --num_hare ${num_hare} --num_stags ${num_stags} --use_feature_normalization \
    --episode_length 200 --use_soft_update  --hard_update_interval_episode 200 --num_env_steps 2000000 --n_training_threads 2  --cuda \
    --msg_iterations 4  --use_dyn_graph --adj_output_dim 32  --eval_interval 2000 --num_eval_episodes 10 --buffer_size 5000 --miscapture_punishment -1.5 \
    --reward_time 0 --highest_orders 3 --batch_size 32 --lr 1e-3 --adj_lr 1e-8 --train_interval_episode 4 --gamma 0.98 --use_wandb --use_linear_lr_decay \
    --entropy_coef 0 --capture_freezes --use_reward_normalization --use_vfunction --num_rank 1 --sparsity 0.3  --gain 0.01 --adj_begin_step 2000000 --gae_lambda 0.97 \
done

