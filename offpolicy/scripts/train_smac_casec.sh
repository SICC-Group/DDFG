#!/bin/sh
env="StarCraft2"
map="MMM2"
algo="casec"
exp="debug"
name="syc"
seed_max=215
seed_min=215
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python train/train_smac.py --env_name ${env} \
     --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
      --seed ${seed} --n_training_threads 2 --buffer_size 5000 --lr 1e-3 --batch_size 8 --use_soft_update \
       --hard_update_interval_episode 200 --num_env_steps 2000000 \
       --log_interval 3000 --eval_interval 20000 --user_name ${name}\
       --use_global_all_local_state  --lamda 0 --msg_iterations 4 --use_wandb --use_save --threshold 0.3 --independent_p_q  
    echo "training is done!"
    #--use_dyn_graph --use_eval --use_wandb
done
# smac threshold=0.3 use_action_repr=True construction_q_var=True q_var_loss=True independent_p_q=False

