import sys
import torch
import numpy as np




# 对比损失
def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    # p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    # p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()
    return loss


def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)     # 使用unsqueeze函数配合*乘法来计算联合概率矩阵
    p_i_j = p_i_j.sum(dim=0)            # n个样本就有n个p矩阵，对n个样本求和，看总体
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

# 弃用
# def mutual_info(view1, view2, EPS=sys.float_info.epsilon):
#     """Mutual  infomation"""
#     _, k = view1.size()
#     p_i_j = compute_joint(view1, view2)
#     assert (p_i_j.size() == (k, k))

#     # p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
#     # p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)
#     p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
#     p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

#     # 下溢处理
#     p_i_j[(p_i_j < EPS).data] = EPS
#     p_j[(p_j < EPS).data] = EPS
#     p_i[(p_i < EPS).data] = EPS

#     mutual_info = p_i_j * (torch.log(p_i_j) \
#                       -  torch.log(p_j) \
#                       -  torch.log(p_i))

#     mutual_info = mutual_info.sum()

#     return mutual_info

# 弃用
# def mutual_info_numpy(view1, view2, EPS=sys.float_info.epsilon):
#     """Mutual  infomation"""
#     # numpy转tensor
#     view1 = torch.from_numpy(view1)
#     view2 = torch.from_numpy(view2)

#     _, k = view1.size()
#     p_i_j = compute_joint(view1, view2)
#     assert (p_i_j.size() == (k, k))

#     # p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
#     # p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)
#     p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
#     p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

#     # 下溢处理
#     p_i_j[(p_i_j < EPS).data] = EPS
#     p_j[(p_j < EPS).data] = EPS
#     p_i[(p_i < EPS).data] = EPS

#     mutual_info = p_i_j * (torch.log(p_i_j) \
#                       -  torch.log(p_j) \
#                       -  torch.log(p_i))

#     mutual_info = mutual_info.sum()
#     mutual_info = mutual_info.numpy()
#     return mutual_info

# 弃用
# def H_12(view1, view2, EPS=sys.float_info.epsilon):
#     """计算两个视角的熵的和"""
#     _, k = view1.size()
#     p_i_j = compute_joint(view1, view2)
#     assert (p_i_j.size() == (k, k))

#     p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
#     p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

#     p_i_j[(p_i_j < EPS).data] = EPS
#     p_j[(p_j < EPS).data] = EPS
#     p_i[(p_i < EPS).data] = EPS

#     H = - p_i_j * ( torch.log(p_j)+torch.log(p_i))

#     H = H.sum()

#     return H

if __name__=='__main__':
    a1 = np.array([[0.1,0.2,0.1,0.5,0.1],[0.2,0.2,0.1,0.2,0.3]])
    a2 = np.array([[0.2,0.1,0.3,0.2,0.2],[0.4,0.1,0.1,0.2,0.2]])
    # view1 = torch.ones([4,10])
    # view2 = 2*torch.ones([4,10])
    view1  = torch.from_numpy(a1)
    view2  = torch.from_numpy(a2)
    print(view1)
    print(view2)
    loss = crossview_contrastive_Loss(view1,view2)
    mutual_info = mutual_info(view1,view2)
    H = H_12(view1,view2)
    print('loss_cl:',loss)
    print('mutual info:',mutual_info)
    print('H:',H)
    print(-9*H-mutual_info)