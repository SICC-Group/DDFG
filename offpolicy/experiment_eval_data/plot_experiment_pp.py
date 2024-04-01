import pandas as pd
import os
import glob
import numpy as np
import sys
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import math
path_vdn = r'.\Predator_prey\p=-1.5_r=0\vdn\run{}'
path_qmix = r'.\Predator_prey\p=-1.5_r=0\qmix\run{}'
path_cw_qmix = r'.\Predator_prey\p=-1.5_r=0\cw-qmix\run{}'
path_ow_qmix = r'.\Predator_prey\p=-1.5_r=0\ow-qmix\run{}'
path_maddpg = r'.\Predator_prey\p=-1.5_r=0\maddpg\run{}'
path_qtran = r'.\Predator_prey\p=-1.5_r=0\qtran\run{}'
path_qplex = r'.\Predator_prey\p=-1.5_r=0\qplex\run{}'
path_dcg = r'.\Predator_prey\p=-1.5_r=0\dcg\run{}'
path_sopcg = r'.\Predator_prey\p=-1.5_r=0\sopcg\run{}'
path_casec = r'.\Predator_prey\p=-1.5_r=0\casec\run{}'
path_ddfg = r'.\Predator_prey\p=-1.5_r=0\ddfg\run{}'

# path_vdn = r'.\Predator_prey\p=-1_r=0\vdn\run{}'
# path_qmix = r'.\Predator_prey\p=-1_r=0\qmix\run{}'
# path_cw_qmix = r'.\Predator_prey\p=-1_r=0\cw-qmix\run{}'
# path_ow_qmix = r'.\Predator_prey\p=-1_r=0\ow-qmix\run{}'
# path_maddpg = r'.\Predator_prey\p=-1_r=0\maddpg\run{}'
# path_qtran = r'.\Predator_prey\p=-1_r=0\qtran\run{}'
# path_qplex = r'.\Predator_prey\p=-1_r=0\qplex\run{}'
# path_dcg = r'.\Predator_prey\p=-1_r=0\dcg\run{}'
# path_sopcg = r'.\Predator_prey\p=-1_r=0\sopcg\run{}'
# path_casec = r'.\Predator_prey\p=-1_r=0\casec\run{}'
# path_ddfg = r'.\Predator_prey\p=-1_r=0\ddfg\run{}'

# path_vdn = r'.\Predator_prey\p=-0.5_r=0\vdn\run{}'
# path_qmix = r'.\Predator_prey\p=-0.5_r=0\qmix\run{}'
# path_cw_qmix = r'.\Predator_prey\p=-0.5_r=0\cw-qmix\run{}'
# path_ow_qmix = r'.\Predator_prey\p=-0.5_r=0\ow-qmix\run{}'
# path_maddpg = r'.\Predator_prey\p=-0.5_r=0\maddpg\run{}'
# path_qtran = r'.\Predator_prey\p=-0.5_r=0\qtran\run{}'
# path_qplex = r'.\Predator_prey\p=-0.5_r=0\qplex\run{}'
# path_dcg = r'.\Predator_prey\p=-0.5_r=0\dcg\run{}'
# path_sopcg = r'.\Predator_prey\p=-0.5_r=0\sopcg\run{}'
# path_casec = r'.\Predator_prey\p=-0.5_r=0\casec\run{}'
# path_ddfg = r'.\Predator_prey\p=-0.5_r=0\ddfg\run{}'

# path_vdn = r'.\Predator_prey\p=0_r=0\vdn\run{}'
# path_qmix = r'.\Predator_prey\p=0_r=0\qmix\run{}'
# path_cw_qmix = r'.\Predator_prey\p=0_r=0\cw-qmix\run{}'
# path_ow_qmix = r'.\Predator_prey\p=0_r=0\ow-qmix\run{}'
# path_maddpg = r'.\Predator_prey\p=0_r=0\maddpg\run{}'
# path_qtran = r'.\Predator_prey\p=0_r=0\qtran\run{}'
# path_qplex = r'.\Predator_prey\p=0_r=0\qplex\run{}'
# path_dcg = r'.\Predator_prey\p=0_r=0\dcg\run{}'
# path_sopcg = r'.\Predator_prey\p=0_r=0\sopcg\run{}'
# path_casec = r'.\Predator_prey\p=0_r=0\casec\run{}'
# path_ddfg = r'.\Predator_prey\p=0_r=0\ddfg\run{}'

# path_vdn = r'.\Predator_prey\p=-1_r=-0.1\vdn\run{}'
# path_qmix = r'.\Predator_prey\p=-1_r=-0.1\qmix\run{}'
# path_cw_qmix = r'.\Predator_prey\p=-1_r=-0.1\cw-qmix\run{}'
# path_ow_qmix = r'.\Predator_prey\p=-1_r=-0.1\ow-qmix\run{}'
# path_maddpg = r'.\Predator_prey\p=-1_r=-0.1\maddpg\run{}'
# path_qtran = r'.\Predator_prey\p=-1_r=-0.1\qtran\run{}'
# path_qplex = r'.\Predator_prey\p=-1_r=-0.1\qplex\run{}'
# path_dcg = r'.\Predator_prey\p=-1_r=-0.1\dcg\run{}'
# path_sopcg = r'.\Predator_prey\p=-1_r=-0.1\sopcg\run{}'
# path_casec = r'.\Predator_prey\p=-1_r=-0.1\casec\run{}'
# path_ddfg = r'.\Predator_prey\p=-1_r=-0.1\ddfg\run{}'

# path_vdn = r'.\Predator_prey\p=-0.5_r=-0.1\vdn\run{}'
# path_qmix = r'.\Predator_prey\p=-0.5_r=-0.1\qmix\run{}'
# path_cw_qmix = r'.\Predator_prey\p=-0.5_r=-0.1\cw-qmix\run{}'
# path_ow_qmix = r'.\Predator_prey\p=-0.5_r=-0.1\ow-qmix\run{}'
# path_maddpg = r'.\Predator_prey\p=-0.5_r=-0.1\maddpg\run{}'
# path_qtran = r'.\Predator_prey\p=-0.5_r=-0.1\qtran\run{}'
# path_qplex = r'.\Predator_prey\p=-0.5_r=-0.1\qplex\run{}'
# path_dcg = r'.\Predator_prey\p=-0.5_r=-0.1\dcg\run{}'
# path_sopcg = r'.\Predator_prey\p=-0.5_r=-0.1\sopcg\run{}'
# path_casec = r'.\Predator_prey\p=-0.5_r=-0.1\casec\run{}'
# path_ddfg = r'.\Predator_prey\p=-0.5_r=-0.1\ddfg\run{}'

def read_data(path,idx,len):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    step1 = []
    reward = []
    df = pd.read_csv(csv_files[idx])
    step1 = np.append(step1, np.array(df["step"]))
    reward = np.append(reward, np.array(df["reward"]))
    return np.array(step1)[:len],np.array(reward)[:len]

if __name__ == '__main__':
    

    step_all = []
    r_vdn = []
    r_qmix = []
    r_cw_qmix = []
    r_ow_qmix = []
    r_maddpg = []
    r_qtran = []
    r_qplex = []
    r_dcg = []
    r_sopcg = []
    r_casec = []
    r_ddfg = []
    l = 979
    idx = 0
    num = 979
    for i in range(1,6):
       step,reward_vdn = read_data(path_vdn.format(i),idx,l)
       step1,reward_qmix = read_data(path_qmix.format(i),idx,l) 
       step2,reward_cw_qmix = read_data(path_cw_qmix.format(i),idx,l) 
       step3,reward_ow_qmix = read_data(path_ow_qmix.format(i),idx,l) 
       step1,reward_maddpg = read_data(path_maddpg.format(i),idx,l) 
       step_qtran,reward_qtran = read_data(path_qtran.format(i),idx,l) 
       step_1,reward_qplex = read_data(path_qplex.format(i),idx,l) 
       step1,reward_dcg = read_data(path_dcg.format(i),idx,l) 
       step1,reward_sopcg = read_data(path_sopcg.format(i),idx,l) 
       step1,reward_casec = read_data(path_casec.format(i),idx,l) 
       step_ddfg,reward_ddfg = read_data(path_ddfg.format(i),idx,l) 
       r_vdn.append(reward_vdn[:num])
       r_qmix.append(reward_qmix[:num])
       r_cw_qmix.append(reward_cw_qmix[:num])
       r_ow_qmix.append(reward_ow_qmix[:num])
       r_maddpg.append(reward_maddpg[:num])
       r_qtran.append(reward_qtran[:num])
       r_qplex.append(reward_qplex[:num])
       r_dcg.append(reward_dcg[:num])
       r_sopcg.append(reward_sopcg[:num])
       r_casec.append(reward_casec[:num])
       r_ddfg.append(reward_ddfg[:num])

    step_all = np.arange(6600,2000000,2000)[:num]
    r_vdn = np.stack(r_vdn,axis=1)
    r_qmix = np.stack(r_qmix,axis=1)
    r_cw_qmix = np.stack(r_cw_qmix,axis=1)
    r_ow_qmix = np.stack(r_ow_qmix,axis=1)
    r_maddpg = np.stack(r_maddpg,axis=1)
    r_qtran = np.stack(r_qtran,axis=1)
    r_qplex = np.stack(r_qplex,axis=1)
    r_dcg = np.stack(r_dcg,axis=1)
    r_sopcg = np.stack(r_sopcg,axis=1)
    r_casec = np.stack(r_casec,axis=1)
    r_ddfg = np.stack(r_ddfg,axis=1)
    r_all = []
    r_max = []
    r_min = []
    r_std = []
    r_all.append(r_vdn)
    r_all.append(r_qmix)
    r_all.append(r_cw_qmix)
    r_all.append(r_ow_qmix)
    r_all.append(r_maddpg)
    r_all.append(r_qtran)
    r_all.append(r_qplex)
    r_all.append(r_dcg)
    r_all.append(r_sopcg)
    r_all.append(r_casec)
    r_all.append(r_ddfg)
    r_all_c = np.zeros((11,l//3+1,5))
    r_all_dcg = np.zeros((l//3+1,5))

    for i in range(len(r_all)):
        for j in range(l//3+1):
            r_all_c[i][j] = np.mean(r_all[i][j*3:j*3+2],axis=0)
    
    tmp = []  
    for i in range(len(r_all)):
        r_std.append(np.std(r_all_c[i],axis=1)/math.sqrt(r_all_c.shape[2]))
        tmp.append(np.mean(r_all_c[i],axis=1))
        r_all[i] = np.mean(r_all[i],axis=1)
            

    color_list = ['blue','orange','sienna','gray','lime','gold','fuchsia','purple','teal','k','red']
    label_list = ['VDN','QMIX','CW-QMIX','OW-QMIX','MADDPG','QTRAN','QPLEX','DCG','SOPCG','CASEC','DDFG']
    color_list_1 = ['lightsteelblue','yellow','lightgreen','cyan','violet','lightcoral']
    x_list = ['0m','0.4m','0.8m','1.2m','1.6m','2m']
    y_list = ['-60','-40','-20','0','20','40','60']
    ax = plt.gca()
    
    ax.spines['top'].set_visible(False) #去掉上边框
    #ax.spines['bottom'].set_visible(False) #去掉下边框
    #ax.spines['left'].set_visible(False) #去掉左边框
    ax.spines['right'].set_visible(False) #去掉右边框
    ax.spines['left'].set_position(('data',0))
    ax.spines['right'].set_position(('data',2e6))

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    idxx = np.arange(0,l,3) #
    plt.figure(1)
    for i in range(len(r_all)):
        plt.plot(step_all[idxx], tmp[i], color=color_list[i], label=label_list[i],linewidth=1.0)
        plt.fill_between(step_all[idxx], tmp[i] - r_std[i], tmp[i] + r_std[i], color=color_list[i], alpha=0.15)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlabel("Environments Steps")
    plt.ylabel("Test Episode Return")
    plt.xticks(np.arange(0, 2e6+1, step=4e5),x_list)
    plt.yticks(np.arange(-60, 61, step=20),y_list)
    plt.grid(linestyle='dotted')
    plt.xlim(0,2e6)
    plt.ylim(-60,60)
    #plt.title("b) punishment p=-1 ${r}_{t}$=-0.1")
    #plt.title("a) no punishment p=0 ${r}_{t}$=0")
    plt.legend(loc=0,ncol=2)

    plt.show()

