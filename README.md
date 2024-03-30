<<<<<<< HEAD
# Dynamic Deep Factor Graph

## Algorithms supported:
- QMIX (MLP and RNN)
- VDN (MLP and RNN)
- MADDPG (MLP and RNN)
- QTRAN (RNN)
- QPLEX (RNN)
- DCG (RNN)
- SOPCG (RNN)
- CASEC (RNN)
- DDFG (RNN)

## Environments supported:

- [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)
- [Multiagent Particle-World Environments (MPEs)](https://github.com/openai/multiagent-particle-envs)
- [predator-prey](https://github.com/wendelinboehmer/dcg)

## 1. Usage
**WARNING: by default all experiments assume a shared policy by all agents i.e. there is one neural network shared by all agents**

* The envs/ subfolder contains environment wrapper implementations for the Predator-prey and SMAC. 

* Code to perform training rollouts and policy updates are contained within the runner/ folder - there is a runner for 
each environment. 

* Executable scripts for training with default hyperparameters can be found in the scripts/ folder. The files are named
in the following manner: train_algo_environment.sh. Within each file, the map name (in the case of SMAC and the Predator-prey) can be altered. 
* Python training scripts for each environment can be found in the scripts/train/ folder. 

* The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones
used in the paper; however, please refer to the appendix for a full list of hyperparameters used. 


## 2. Installation

 Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# install on-policy package
cd off-policy
pip install -e .
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### 2.1 Install StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

* download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

* To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.


### 2.2 Install MPE

``` Bash
# install this package first
pip install seaborn
```

There are 3 Cooperative scenarios in MPE:

* simple_spread
* simple_speaker_listener, which is 'Comm' scenario in paper
* simple_reference

## 3.Train
Here we use train_mpe_maddpg.sh as an example:
```
cd offpolicy/scripts
chmod +x ./train_mpe_maddpg.sh
./train_mpe_maddpg.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 



=======
# DDFG
>>>>>>> a6d29e56d8596b6ce40895418c459e9acf7b163b
