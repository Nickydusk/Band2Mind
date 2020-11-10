import os
if 'data' not in os.listdir():
    os.mkdir('data')
os.chdir('data')
if 'train' not in os.listdir():
    os.mkdir('train')
os.chdir('../')

import pandas as pd
import numpy as np
import random

N_features = 20
N_batch = 1000
store_path = 'data/train/'

def insert_random_feature(df,c_name,shift,scale):
    '''
    给dataframe插入一列服从N(shift,scale^2)的随机数据
    :parm df: 要操作的dataframe
    :parm c_name: 插入列名
    :parm shift: 正态分布的均值
    :parm scale: 正态分布的方差
    '''

    inlist = np.random.randn(df.shape[0])
    inlist *= scale
    inlist += shift

    df[c_name] = inlist
def get_label(feature_list):
    '''
    按照某策略，根据特征值生成label，加入一定随机性
    :parm feature_list: 特征列表，其中每一项为某高斯分布的（均值，方差）
    '''

    for i in range(1,len(feature_list)):
        feature_list[0] += feature_list[i]
    
    pan = (feature_list[0][0] + feature_list[0][1])/len(feature_list)
    if  pan + random.randint(0,1) >= 7.5:
        ret =  1
    else:
        ret = -1
    
    if random.randint(1,100) <= 5:
        ret *= -1
    
    return ret
    

if __name__=='__main__':
    for r in range(N_batch):
        dfa = pd.bdate_range(start='20201108',end='20201110',freq='1min')
        dfb = pd.DataFrame(dfa,columns=['time'])

        feature_list = list()

        for i in range(N_features-1):
            a,b = random.randint(-100,100),random.randint(1,100)
            insert_random_feature(dfb,'feature_%d'%i,a,b)
            feature_list.append([a,b])
        
        label = get_label(feature_list)
        dfb.to_csv(store_path+"train_%d_%d.csv"%(r,label),index=False)
        print(r)