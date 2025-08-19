import sys
import os
import numpy as np
from pathlib import Path
import wandb
import socket
import setproctitle
import torch
import math
from gym.spaces import Discrete
from offpolicy.config import get_config
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from offpolicy.envs.hmpe.hmpe import hmpe
from offpolicy.envs.starcraft2.smac_maps import get_map_params
from offpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from offpolicy.runner.mlp.hmpe_runner import HMPERunner as Runner
import pandas as pd



def parse_args(args, parser):
    parser.add_argument('--use_available_actions', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--use_global_all_local_state', action='store_true',
                        default=False, help="Whether to use available actions")
    parser.add_argument('--num_factor', type=int,
                        default=28, help="number of factor")
    parser.add_argument('--num_agents', type=int,
                        default=0, help="number of factor")
    parser.add_argument('--highest_orders', type=int,
                        default=2, help="number of agents")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        # init wandb
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                 1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # set seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    env = hmpe(max_cycles = all_args.episode_length, is_goaltrain = False)

    # create policies and mapping fn
    if all_args.share_policy:
        #print(env.share_observation_space[0])
        policy_info = {
            'policy_0': {"cent_obs_dim": env.state_dim,
                         "cent_act_dim": None,
                         "obs_space": [env.obs_dim],
                         "share_obs_space": [env.state_dim],
                         "act_space": Discrete(env.action_space),
                         "unit_dim": None}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(env.num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)
    eval_env = env
    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "num_agents": env.num_agents,
              "device": device,
              "run_dir": run_dir,
              "use_same_share_obs": all_args.use_same_share_obs,
              "use_available_actions": all_args.use_available_actions,
              "eval_env":eval_env}

    
    progress_filename = os.path.join(run_dir,'config.csv')
    df = pd.DataFrame(list(all_args.__dict__.items()),columns=['Name', 'Value'])
    df.to_csv(progress_filename,index=False)
    
    progress_filename = os.path.join(run_dir,'progress.csv')
    df = pd.DataFrame(columns=['step','reward','num_put_trash'])
    df.to_csv(progress_filename,index=False)
    
    progress_filename = os.path.join(run_dir,'progress_eval.csv')
    df = pd.DataFrame(columns=['step','reward','num_put_trash'])
    df.to_csv(progress_filename,index=False)
    
    progress_filename_train = os.path.join(run_dir,'progress_train.csv')
    df = pd.DataFrame(columns=['step','loss','Q_tot']) 
    df.to_csv(progress_filename_train,index=False)
    
    progress_filename_train = os.path.join(run_dir,'progress_train_adj.csv')
    df = pd.DataFrame(columns=['step','advantage','clamp_ratio','rl_loss','auto_loss']) 
    df.to_csv(progress_filename_train,index=False)
    total_num_steps = 0
    runner = Runner(config=config)
    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()

    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
