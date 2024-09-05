import logging
import numpy as np
import math
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

# def next_batch(X1, X2, batch_size):
def next_batch(X_list, batch_size):
    """Return data for next batch"""
    tot = X_list[0].shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_xv = []
        for v in range(len(X_list)):
            batch_xv.append(X_list[v][start_idx: end_idx, ...])
        # batch_x1 = X1[start_idx: end_idx, ...]
        # batch_x2 = X2[start_idx: end_idx, ...]
        yield (batch_xv, (i + 1))


# 弃用
def cal_std(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        logger.info('ACC:'+ str(arg[0]))
        logger.info('NMI:'+ str(arg[1]))
        logger.info('ARI:'+ str(arg[2]))
        output = """ ACC {:.2f} std {:.2f} NMI {:.2f} std {:.2f} ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100,
                                                                                                 np.std(arg[0]) * 100,
                                                                                                 np.mean(arg[1]) * 100,
                                                                                                 np.std(arg[1]) * 100,
                                                                                                 np.mean(arg[2]) * 100,
                                                                                                 np.std(arg[2]) * 100)
    elif len(arg) == 1:
        logger.info(arg)
        output = """auc {:.2f} std {:.2f}""".format(np.mean(arg), np.std(arg))
    logger.info(output)

    return

# 归一化到 [0,1]
def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


# 计算auc
def compute_auc(labels, scores):
    return roc_auc_score(labels, scores)

# 根据离群点比例计算 得分阈值per
def get_percentile(scores, threshold):
    per = np.percentile(scores, 100 - int(100 * threshold))
    return per

# 计算f1等准确度
def compute_f1_score(labels, scores, outlier_rate):
    '''
        args: 
            labels: 标签
            scores: 各样本点的得分 
            outlier_rate: 离群点比例
    '''
    per = get_percentile(scores, outlier_rate)
    y_pred = (scores >= per)
    # print(np.sum(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(labels.astype(int),
                                                               y_pred.astype(int),
                                                               average='binary')
    return precision, recall, f1


