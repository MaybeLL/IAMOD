'''
离群得分函数
'''

from sklearn import metrics
import numpy as np
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
import sys
from munkres import Munkres
from sklearn.metrics import roc_auc_score
from loss import *


# 归一化
def cal_final_score(dualpre_scores,recon_scores, knn_scores,mi_score, config):

    s0 = (dualpre_scores - np.min(dualpre_scores)) / (np.max(dualpre_scores) - np.min(dualpre_scores))
    s1 = (recon_scores - np.min(recon_scores)) / (np.max(recon_scores) - np.min(recon_scores))
    s2 = (knn_scores - np.min(knn_scores)) / (np.max(knn_scores) - np.min(knn_scores))
    dual_score= config['training']['dualpre_scores_weight']*s0
    recon_score = config['training']['recon_scores_weight']*s1
    knn_score = config['training']['knn_scores_weight']*s2

    total_scores = dual_score + recon_score + knn_score    

    return total_scores


# 返回最终的离群点得分
def cal_scores(all_view_encoded, num_neibs, all_recon_error,all_dualpre_error,config):
    '''
    功能：计算离群点得分
    '''
    from sklearn.neighbors import NearestNeighbors
    num_views = len(all_view_encoded)
    num_obj = all_view_encoded[0].shape[0]

    # -----------重构得分（弃用）-----------------
    # recon_scores = np.zeros(num_obj)
    # for i in range(num_views):
    #     s1 = (all_recon_error[i] - np.min(all_recon_error[i])) / (np.max(all_recon_error[i]) - np.min(all_recon_error[i]))
    #     recon_scores += s1


    # ----------------跨视角一致性得分---------------------
    dualpre_scores = np.zeros(num_obj)
    for i in range(num_views):
        s2 = (all_dualpre_error[i] - np.min(all_dualpre_error[i])) / (np.max(all_dualpre_error[i]) - np.min(all_dualpre_error[i]))
        dualpre_scores += s2

    # -----------------邻居一致性得分------------------------
    encoded_concen = np.concatenate(all_view_encoded,axis=1)
    neibs = []
    NNK = NearestNeighbors(n_neighbors=num_neibs+1, algorithm='ball_tree')


    nbrs = NNK.fit(encoded_concen)
    distances, indices = nbrs.kneighbors(encoded_concen)   
    knn_scores = np.sum(distances, axis=1)
    tmp = list(indices[0][1:])


    # ------------------- MI(inlier,outlier) 邻居互信息得分 （论文中未使用，请注释）----------------
    # mi_score = np.zeros(num_obj)

    # for i in range(num_obj):
    #     # 找邻居表征
    #     nei = indices[i][1:]
    #     Zn = encoded_concen[nei]
    #     # 复制k行
    #     z_i = encoded_concen[i][np.newaxis,:]
    #     Zi = np.repeat(z_i,num_neibs,axis=0)
        
    #     mi_score[i] = mutual_info_numpy(Zi,Zn)
    # # 归一化
    # mi_score = (mi_score - np.min(mi_score)) / (np.max(mi_score)- np.min(mi_score))
    

    score = cal_final_score(dualpre_scores,recon_scores,knn_scores,mi_score,config)
    return score






