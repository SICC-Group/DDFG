#!/bin/sh
env="StarCraft2"
map="8m_vs_9m"
algo="rddfg_cent_rw"
exp="final"
name="syc"
seed_max=1378
seed_min=1378

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python train/train_smac.py --env_name ${env} \
     --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
      --seed ${seed} --n_training_threads 4 --buffer_size 5000 --batch_size 32 --use_soft_update \
      --hard_update_interval_episode 200 --num_env_steps 2000000 \
      --log_interval 3000 --eval_interval 20000 --user_name ${name}\
      --msg_iterations 4 --adj_output_dim 64 --highest_orders 2 \
      --lr 5e-4 --critic_lr 5e-4 --adj_lr 1e-8 --entropy_coef 0 --capture_freezes --use_wandb \
      --use_vfunction --use_global_all_local_state --gamma 0.97 --num_rank 1 \
      --train_interval_episode 4 --train_adj_episode 4 --adj_buffer_size 4 --num_mini_batch 1 \
      --sparsity 0.3 --gain 0.01 --adj_begin_step 0 --gae_lambda 0.95 --use_linear_lr_decay --use_dyn_graph
done


