import pandas as pd
import os
import glob
import numpy as np
import sys
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import math
path_vdn = r'.\smac\3s5z\vdn\run{}'
path_qmix = r'.\smac\3s5z\qmix\run{}'
path_cw_qmix = r'.\smac\3s5z\cw-qmix\run{}'
path_ow_qmix = r'.\smac\3s5z\ow-qmix\run{}'
path_maddpg = r'.\smac\3s5z\maddpg\run{}'
path_qtran = r'.\smac\3s5z\qtran\run{}'
path_qplex = r'.\smac\3s5z\qplex\run{}'
path_dcg = r'.\smac\3s5z\dcg\run{}'
path_sopcg = r'.\smac\3s5z\sopcg\run{}'
path_casec = r'.\smac\3s5z\casec\run{}'
path_ddfg = r'.\smac\3s5z\ddfg\run{}'

# path_vdn = r'.\smac\5m_vs_6m\vdn\run{}'
# path_qmix = r'.\smac\5m_vs_6m\qmix\run{}'
# path_cw_qmix = r'.\smac\5m_vs_6m\cw-qmix\run{}'
# path_ow_qmix = r'.\smac\5m_vs_6m\ow-qmix\run{}'
# path_maddpg = r'.\smac\5m_vs_6m\maddpg\run{}'
# path_qtran = r'.\smac\5m_vs_6m\qtran\run{}'
# path_qplex = r'.\smac\5m_vs_6m\qplex\run{}'
# path_dcg = r'.\smac\5m_vs_6m\dcg\run{}'
# path_sopcg = r'.\smac\5m_vs_6m\sopcg\run{}'
# path_casec = r'.\smac\5m_vs_6m\casec\run{}'
# path_ddfg = r'.\smac\5m_vs_6m\ddfg\run{}' 

# path_vdn = r'.\smac\8m_vs_9m\vdn\run{}'
# path_qmix = r'.\smac\8m_vs_9m\qmix\run{}'
# path_cw_qmix = r'.\smac\8m_vs_9m\cw-qmix\run{}'
# path_ow_qmix = r'.\smac\8m_vs_9m\ow-qmix\run{}'
# path_maddpg = r'.\smac\8m_vs_9m\maddpg\run{}'
# path_qtran = r'.\smac\8m_vs_9m\qtran\run{}'
# path_qplex = r'.\smac\8m_vs_9m\qplex\run{}'
# path_dcg = r'.\smac\8m_vs_9m\dcg\run{}'
# path_sopcg = r'.\smac\8m_vs_9m\sopcg\run{}'
# path_casec = r'.\smac\8m_vs_9m\casec\run{}'
# path_ddfg = r'.\smac\8m_vs_9m\ddfg\run{}'

# path_vdn = r'.\smac\MMM2\vdn\run{}'
# path_qmix = r'.\smac\MMM2\qmix\run{}'
# path_cw_qmix = r'.\smac\MMM2\cw-qmix\run{}'
# path_ow_qmix = r'.\smac\MMM2\ow-qmix\run{}'
# path_maddpg = r'.\smac\MMM2\maddpg\run{}'
# path_qtran = r'.\smac\MMM2\qtran\run{}'
# path_qplex = r'.\smac\MMM2\qplex\run{}'
# path_dcg = r'.\smac\MMM2\dcg\run{}'
# path_sopcg = r'.\smac\MMM2\sopcg\run{}'
# path_casec = r'.\smac\MMM2\casec\run{}'
# path_ddfg = r'.\smac\MMM2\ddfg\run{}'

# path_vdn = r'.\smac\1c3s5z\vdn\run{}'
# path_qmix = r'.\smac\1c3s5z\qmix\run{}'
# path_cw_qmix = r'.\smac\1c3s5z\cw-qmix\run{}'
# path_ow_qmix = r'.\smac\1c3s5z\ow-qmix\run{}'
# path_maddpg = r'.\smac\1c3s5z\maddpg\run{}'
# path_qtran = r'.\smac\1c3s5z\qtran\run{}'
# path_qplex = r'.\smac\1c3s5z\qplex\run{}'
# path_dcg = r'.\smac\1c3s5z\dcg\run{}'
# path_sopcg = r'.\smac\1c3s5z\sopcg\run{}'
# path_casec = r'.\smac\1c3s5z\casec\run{}'
# path_ddfg = r'.\smac\1c3s5z\ddfg\run{}'

def read_data_1(path,idx,len):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    step1 = []
    reward = []
    df = pd.read_csv(csv_files[idx])
    step1 = np.append(step1, np.array(df["step"]))
    reward = np.append(reward, np.array(df["win_rate"]))


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
    l = 100
    idx = 0
    num = 99
    for i in range(1,6):
        step,reward_vdn = read_data_1(path_vdn.format(i),idx,l)
        step1,reward_qmix = read_data_1(path_qmix.format(i),idx,l)
        step1,reward_cw_qmix = read_data_1(path_cw_qmix.format(i),idx,l) 
        step1,reward_ow_qmix = read_data_1(path_ow_qmix.format(i),idx,l)  
        step1,reward_maddpg = read_data_1(path_maddpg.format(i),idx,l) 
        step_qtran,reward_qtran = read_data_1(path_qtran.format(i),idx,l) 
        step1,reward_qplex = read_data_1(path_qplex.format(i),idx,l) 
        step1,reward_dcg = read_data_1(path_dcg.format(i),idx,l) 
        step1,reward_sopcg = read_data_1(path_sopcg.format(i),idx,l) 
        step1,reward_casec = read_data_1(path_casec.format(i),idx,l) 
        step1,reward_ddfg = read_data_1(path_ddfg.format(i),idx,l) 
        step_all.append(step[:num])
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
    r_all_c = np.zeros((11,l-1,5))

    for i in range(len(r_all)):
        for j in range(l-1):
            if j>=3:
                r_all_c[i][j] = np.mean(r_all[i][j-3:j+3],axis=0)
            else:
                r_all_c[i][j] = np.mean(r_all[i][0:j+3],axis=0)
    
    tmp = []  
    for i in range(len(r_all)):
        r_std.append(np.std(r_all_c[i],axis=1)/math.sqrt(r_all_c.shape[2]))
        tmp.append(np.mean(r_all_c[i],axis=1))
        r_all[i] = np.mean(r_all[i],axis=1)

    color_list = ['blue','orange','sienna','gray','lime','gold','fuchsia','purple','teal','k','red']
    label_list = ['VDN','QMIX','CW-QMIX','OW-QMIX','MADDPG','QTRAN','QPLEX','DCG','SOPCG','CASEC','DDFG']
    x_list = ['0m','0.4m','0.8m','1.2m','1.6m','2m']
    y_list = ['0','20','40','60','80','100']
    ax = plt.gca()
    
    ax.spines['top'].set_visible(False) #去掉上边框
    #ax.spines['bottom'].set_visible(False) #去掉下边框
    #ax.spines['left'].set_visible(False) #去掉左边框
    ax.spines['right'].set_visible(False) #去掉右边框
    ax.spines['left'].set_position(('data',0))
    ax.spines['right'].set_position(('data',2e6))
    #移位置 设为原点相交
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    idxx = np.arange(0,l-1) #np.arange(0,l,3)
    plt.figure(1)
    for i in range(len(r_all)):
        if i != 1:
            plt.plot(step_all[0][idxx], tmp[i], color=color_list[i], label=label_list[i],linewidth=1.0)
            plt.fill_between(step_all[0][idxx], tmp[i] - r_std[i], tmp[i] + r_std[i], color=color_list[i], alpha=0.12)

    plt.xlabel("Environmental Steps")
    plt.ylabel("Median Test Win Rate")
    plt.xticks(np.arange(0, 2e6+1, step=4e5),x_list)
    plt.yticks(np.arange(0, 1.1, step=0.2),y_list)
    plt.grid(True)
    plt.xlim(0,2e6)
    
    plt.ylim(0,1)
    plt.title("(b) 1c3s5z")
    plt.legend(loc=0)

    plt.show()

