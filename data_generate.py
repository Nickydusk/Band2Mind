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
def get_label(p):
    '''
    按照给定概率 返回 p:1 1-p:-1
    '''

    a = np.random.randint(low=0,high=1000)
    if a>1000*p:
        return -1
    else:
        return 1
    

if __name__=='__main__':
    for r in range(1000):
        a = pd.bdate_range(start='20201108',end='20201110',freq='1min')
        b = pd.DataFrame(a,columns=['time'])

        for i in range(N_features-1):
            insert_random_feature(b,'feature_%d'%i,random.randint(-100,100),random.randint(1,100))
        
        label = get_label(0.99)
        b.to_csv(store_path+"train_%d_%d.csv"%(r,label),index=False)
        print(b)