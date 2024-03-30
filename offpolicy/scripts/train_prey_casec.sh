#!/bin/sh
#!/bin/sh
env="disperse"
num_hare=0
num_agents=12
num_stags=6 
num_factor=78
algo="casec"
exp="debug"
name="syc"
seed_max=215
seed_min=215

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_prey.py --user_name ${name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --num_factor ${num_factor} --seed ${seed} --num_hare ${num_hare} --num_stags ${num_stags} --n_training_threads 2 --use_soft_update --lr 5e-4 --hard_update_interval_episode 200 --num_env_steps 2000000  --batch_size 32 --lamda 0 --msg_iterations 4 --miscapture_punishment -0.5 --reward_time -0.1 --buffer_size 5000 --eval_interval 2000 --num_eval_episodes 10  --epsilon_anneal_time 50000 --use_wandb --use_reward_normalization --capture_freezes --use_save --highest_orders 2 --use_action_repr --threshold 0.5 --cuda  --episode_length 10  --num_env_steps 1000000 --catch_reward 10 
    #  --use_wandb  --cuda --independent_p_q 
    echo "training is done!"
done
# pp threshold=0.5 use_action_repr=False construction_q_var=True q_var_loss=True independent_p_q=True
