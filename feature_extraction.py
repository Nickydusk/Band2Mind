import pandas as pd
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

source_path = './data/train/'
store_path = './data/features.csv'
N_PCA = 3


def file2FeatureVector(filename):
    '''
    对文件信息进行特征提取，返回行向量，最后一列为label(-1,1)
    :parm filename: 待处理文件名
    '''

    df = pd.read_csv(filename)
    df = df.iloc[:,1:] # 去除时间列
    
    s_skew = df.skew()
    s_skew = s_skew.rename(lambda x:x+"_skew")
    
    s_kurt = df.kurt()
    s_kurt = s_kurt.rename(lambda x:x+"_kurt")
    
    s_std = df.std()
    s_std = s_std.rename(lambda x:x+"_std")
    
    s_var = df.var()
    s_var = s_var.rename(lambda x:x+"_var")

    
    ret = pd.concat([s_skew,s_kurt,s_std,s_var])
    
    # label加入最后一位
    label = filename.split('_')[-1]
    label = label.split('.')[0]
    label = pd.Series(label,['label'])
    ret = ret.append(label)

    return ret

def make_header():
    '''
    创建空的dataframe，用于后续添加数据
    '''
    fname = source_path + os.listdir(source_path)[0]
    seri = file2FeatureVector(fname)
    ret = pd.DataFrame([seri])
    ret.drop(0,axis=0,inplace=True)
    return ret

def df_process(df):
    '''
    对特征矩阵进行pca降维
    :parm df: 待操作的原始dataframe
    '''
    data = df.iloc[:,:-1].values
    pca = PCA(n_components = N_PCA)
    pca_data = pca.fit_transform(data)
    ret = pd.DataFrame(pca_data,columns=["pca_%d"%x for x in range(N_PCA)])
    ret = pd.concat([ret,df.iloc[:,-1]],axis=1)
    return ret

def show_pca(df):
    '''
    将pca降维后的数据进行可视化展示
    :parm df: 降维后的dataframe，特征<=3
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    
    for type,m in [(1,'o'),(-1,'^')]:
        df_t = df[df['label']==type]
        xs = df_t.iloc[:,0].values
        ys = df_t.iloc[:,1].values
        zs = df_t.iloc[:,2].values
        ax.scatter(xs,ys,zs,marker=m)

    plt.show()



if __name__=='__main__':
    # df = make_header()
    # for filename in os.listdir(source_path):
    #     print(filename)
    #     seri = file2FeatureVector(source_path+filename)
    #     df = df.append(seri,ignore_index=True)

    # df = df_process(df)
    # df.to_csv(store_path,index=False)

    df = pd.read_csv(store_path)
    show_pca(df)