import os, random, sys
from re import S
import numpy as np
import scipy.io as sio
import util
from scipy import sparse

# ---------功能： 读数据文件，构造数据格式：所有视角的特征X_list = [[n,d1],[n,d2]]  标签 Y_list
# 注：数据加载代码请自行根据数据格式完成，只需最终加载的数据格式如上，即可正常跑run.py

def load_data(config):
    """Load data """
    data_name = config['dataset']   
    main_dir = sys.path[0]
    X_list = []
    Y_list = []


    # 各视角维度：48,40,254,1984,512,928
    if data_name in ['caltech7']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/real/caltech7.mat')  
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   # 取任意
            for i in range(6):
                x_view = np.transpose(X[t][0][0][i])
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
        # print(Y_list)
        # 获得各个视角数据
    elif data_name in ['toydataset']:
        data,y  = np.load('/home/ljl/work/mycode/IAMOD/data/toydataset.npy',allow_pickle=True)
        X_list.append(data[0])
        X_list.append(data[1])
        Y_list.append(np.squeeze(y))
    elif data_name in ['awa']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/real/awa.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(6):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['msrc-v1']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/real/msrc-v1.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(5):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['attribute/iris']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Attribute Outlier/iris.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['class/iris']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Class Outlier/iris.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['attribute_class/iris']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Class-Attribute Outlier/iris.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['attribute/pima']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Attribute Outlier/pima.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['class/pima']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Class Outlier/pima.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['attribute_class/pima']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Class-Attribute Outlier/pima.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['attribute/zoo']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Attribute Outlier/zoo.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['class/zoo']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Class Outlier/zoo.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['attribute_class/zoo']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Class-Attribute Outlier/zoo.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['attribute/ionosphere']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Attribute Outlier/ionosphere.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['class/ionosphere']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Class Outlier/ionosphere.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['attribute_class/ionosphere']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Class-Attribute Outlier/ionosphere.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['attribute/letter']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Attribute Outlier/letter.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['class/letter']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Class Outlier/letter.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    elif data_name in ['attribute_class/letter']:
        mat = sio.loadmat('/home/ljl/work/mycode/IAMOD/data/UCI/Class-Attribute Outlier/letter.mat')
        X = mat['X']    
        class_y = mat['class_y']
        y = mat['y']
        for t in [0]:   
            for i in range(X[t][0][0].shape[0]):
                x_view = np.transpose(X[t][0][0][i])    
                x_view = util.normalize(x_view).astype('float32')
                X_list.append(x_view)
            Y_list.append(np.squeeze(np.transpose(y[t][0])))
    else :
        raise Exception('未知数据集')
    return X_list, Y_list


if __name__=='__main__':
    from configure import get_default_config

    config = get_default_config('attribute/leaf')
    config['print_num'] = 100
    config['dataset'] = 'attribute/leaf'
    X_list, Y_list = load_data(config)
    print(X_list)