# ----------------MINE的使用例子------------------------
# *_*coding:utf-8 *_*

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt





def gen_x(num, dim):
    return np.random.normal(0., np.sqrt(SIGNAL_POWER), [num, dim])    # 方差是根号3


def gen_y(x, num, dim):
    return x + np.random.normal(0., np.sqrt(SIGNAL_NOISE), [num, dim])  # 方差是根号0.2


def true_mi(power, noise, dim):
    return dim * 0.5 * np.log2(1 + power/noise)





class MINE(nn.Module):
    def __init__(self, hidden_size=10):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2 * data_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x, ], dim=0)
        idx = torch.randperm(batch_size)
        # shuffle操作来获得边缘分布采样的样本
        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]   # 联合分布
        pred_x_y = logits[batch_size:]  # px乘py
        loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))    # 估计的互信息
        # compute loss, you'd better scale exp to bit
        return loss


if __name__ == "__main__":

    SIGNAL_NOISE = 0.2
    SIGNAL_POWER = 3

    data_dim = 3
    num_instances = 20000

    mi = true_mi(SIGNAL_POWER, SIGNAL_NOISE, data_dim)
    print('True MI:', mi)


    hidden_size = 10
    n_epoch = 500


    model = MINE(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    plot_loss = []
    all_mi = []
    for epoch in tqdm(range(n_epoch)):
        x_sample = gen_x(num_instances, data_dim)
        y_sample = gen_y(x_sample, num_instances, data_dim)

        x_sample = torch.from_numpy(x_sample).float()
        y_sample = torch.from_numpy(y_sample).float()

        loss = model(x_sample, y_sample)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        all_mi.append(-loss.item())


    fig, ax = plt.subplots()
    ax.plot(range(len(all_mi)), all_mi, label='MINE Estimate')
    ax.plot([0, len(all_mi)], [mi, mi], label='True Mutual Information')
    ax.set_xlabel('training steps')
    ax.legend(loc='best')
    plt.show()
