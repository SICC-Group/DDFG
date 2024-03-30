import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import wandb
import socket
import setproctitle
import torch
import math
from offpolicy.config import get_config
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from offpolicy.envs.predator_prey.stag_hunt import StagHunt
from offpolicy.envs.predator_prey.aloha import AlohaEnv
from offpolicy.envs.predator_prey.hallway import HallwayEnv
from offpolicy.envs.predator_prey.sensor import SensorEnv
from offpolicy.envs.predator_prey.gather import GatherEnv
from offpolicy.envs.predator_prey.disperse import DisperseEnv
from offpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Predator_prey":
                env = StagHunt(all_args)
            elif all_args.env_name == "aloha":
                env = AlohaEnv(all_args)
            elif all_args.env_name == "hallway":
                env = HallwayEnv(all_args)
            elif all_args.env_name == "sensor":
                env = SensorEnv(all_args)
            elif all_args.env_name == "gather":
                env = GatherEnv(all_args)
            elif all_args.env_name == "disperse":
                env = DisperseEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Predator_prey":
                env = StagHunt(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--use_available_actions', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--use_global_all_local_state', action='store_true',
                        default=False, help="Whether to use available actions")
    parser.add_argument("--num_stags", type=int, default=3)
    parser.add_argument("--num_hare", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=3, help="number of agents")
    parser.add_argument('--toroidal', action='store_true',
                        default=False, help="whether the world is bounded (False) or toroidal (True)")
    parser.add_argument('--world_shape',type=list, default=[10, 10], help="the shape of the grid-world [height, width]")
    parser.add_argument('--agent_obs',type=list, default=[2,2], help="(radius-1) of the agent's observation, e.g., [0, 0] observes only one pixel")
    parser.add_argument('--agent_move_block',type=list, default=[0,1,2], help="by which entities is an agent's move blocked (0=agents, 1=stags, 2=hare)")
    parser.add_argument('--capture_action_conditions',type=list, default=[3,1], help="number of agents that have to simultaneously execute catch action")
    #parser.add_argument('--capture_action_conditions',nargs = '+', help="number of agents that have to simultaneously execute catch action",required=True)
    parser.add_argument('--reward_hare', type=int,
                        default=1, help="reward for capturing a hare")
    parser.add_argument('--reward_stag', type=int,
                        default=10, help="reward for capturing a stag")
    parser.add_argument('--reward_collision', type=float,
                        default=0, help="reward (or punishment) for colliding with other agents")
    parser.add_argument('--reward_time', type=float,
                        default=0, help="reward (or punishment) given at each time step")
    parser.add_argument('--miscapture_punishment', type=float, default=-2)
    parser.add_argument('--capture_action', action='store_false',
                        default=True, help="whether capturing requires an extra action (True) or just capture_conditions (False)")
    parser.add_argument('--capture_terminal', action='store_true',
                        default=False, help="whether capturing any prey ends the episode (i.e. only one prey can be caught)")
    parser.add_argument('--p_stags_rest', type=float, default=0,help="probability that a stag will not move (at each time step)")
    parser.add_argument('--p_hare_rest', type=float, default=0)
    parser.add_argument('--prevent_cannibalism', action='store_false',
                        default=True, help="If set to False, prey can be captured by other prey (witch is rewarding)")
    parser.add_argument('--remove_frozen', action='store_false',
                        default=True, help="whether frozen agents are removed (True) or still present in the world (False)")
    parser.add_argument('--capture_freezes', action='store_false',
                        default=True, help="whether capturing any prey freezes the participating agents (True) or not (False)")
    parser.add_argument('--num_factor', type=int,
                        default=28, help="number of factor")
    parser.add_argument('--highest_orders', type=int,
                        default=3, help="number of agents")
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
                         dir=str(run_dir),
                         job_type="training",
                         reinit=False)
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

    env = make_train_env(all_args)
    num_agents = all_args.num_agents
    all_args.num_factor = int(math.factorial(num_agents)//(math.factorial(all_args.highest_orders)*math.factorial(num_agents-all_args.highest_orders)) * all_args.sparsity)
    # create policies and mapping fn
    if all_args.share_policy:
        print(env.share_observation_space[0])
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0],
                         "unit_dim": env.envs[0].unit_dim}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)

    # choose algo
    if all_args.algorithm_name in ["rmatd3", "rmaddpg", "rmasac", "qtran","wqmix","qmix", "vdn","qplex","rddfg_cent_rw","rmfg_cent","sopcg","casec"]:
        from offpolicy.runner.rnn.prey_runner import PREYRunner as Runner
        assert all_args.n_rollout_threads == 1, ("only support 1 env in recurrent version.")
        eval_env = make_train_env(all_args)
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn"]:
        from offpolicy.runner.mlp.smac_runner import SMACRunner as Runner
        eval_env = make_eval_env(all_args)
    else:
        raise NotImplementedError
    
    adj = torch.zeros((all_args.num_agents,all_args.num_factor),dtype=torch.int64)
    index = 0
    n = 0
    if all_args.use_dyn_graph == False and all_args.equal_vdn == False and all_args.algorithm_name in ["rddfg_cent_rw","rmfg_cent","sopcg","casec"]:
        for i in range(all_args.num_agents-1):
            for j in range(i+1,all_args.num_agents):
                adj[i,index] = 1
                adj[j,index] = 1
                index = index + 1
        for i in range(index,all_args.num_factor):
            adj[n,i] = 1
            n = n + 1
               
    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": eval_env,
              "num_agents": num_agents,
              "device": device,
              "run_dir": run_dir,
              "use_same_share_obs": all_args.use_same_share_obs,
              "use_available_actions": all_args.use_available_actions,
              "adj": adj}

    total_num_steps = 0
    runner = Runner(config=config)
    
    progress_filename = os.path.join(run_dir,'config.csv')
    df = pd.DataFrame(list(all_args.__dict__.items()),columns=['Name', 'Value'])
    df.to_csv(progress_filename,index=False)
    
    progress_filename = os.path.join(run_dir,'progress.csv')
    df = pd.DataFrame(columns=['step','reward'])
    df.to_csv(progress_filename,index=False)
    
    progress_filename = os.path.join(run_dir,'progress_eval.csv')
    df = pd.DataFrame(columns=['step','reward'])
    df.to_csv(progress_filename,index=False)
    
    progress_filename_train = os.path.join(run_dir,'progress_train.csv')
    df = pd.DataFrame(columns=['step','loss','Q_tot','grad_norm']) 
    df.to_csv(progress_filename_train,index=False)
    
    progress_filename_train = os.path.join(run_dir,'progress_train_adj.csv')
    df = pd.DataFrame(columns=['step','advantage','clamp_ratio','rl_loss','entropy_loss','grad_norm']) 
    df.to_csv(progress_filename_train,index=False)
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
