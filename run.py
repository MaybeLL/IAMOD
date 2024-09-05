import argparse
import collections
import itertools
import torch

from model import IAMOD
# from get_mask import get_mask
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config

dataset = {
    0: "caltech7",
    1: "toydataset",
    2: "awa"
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='100', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='2', help='number of test times')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:1' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    config['num_neibor'] = 5
    logger = get_logger()

    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("  %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    # Load data
    X_list_np, Y_list = load_data(config)
    X_list = [torch.from_numpy(x).float().to(device) for x in X_list_np]


    accumulated_metrics = collections.defaultdict(list)
    
    # 重复几次
    for i in range(1, args.test_time + 1):

        torch.backends.cudnn.deterministic = True

        iamod = IAMOD(config)

        para = []
        for i in range(config['training']['num_view']):
            d = dict(params=iamod.autoencoder_list[i].parameters(),lr=config['training']['lr'])
            para.append(d)
        for i in range(config['training']['num_view']):
            d = dict(params=iamod.prediction_list[i].parameters(),lr=config['training']['lr'])
            para.append(d)
        optimizer = torch.optim.Adam(para)

        iamod.to_device(device)


        # Training 
        auc = iamod.train(config, logger, X_list, Y_list,
                                        optimizer, device)
        accumulated_metrics['auc'].append(auc)


    logger.info('--------------------Training over--------------------')
    cal_std(logger, accumulated_metrics['auc'])


if __name__ == '__main__':
    main()
