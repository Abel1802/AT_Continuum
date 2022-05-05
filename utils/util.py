import logging
import torch


def get_logger(filename, verbosity=1, name=None):
    '''
    '''
    level_dict = {0: logging.DEBUG,
                  1: logging.INFO,
                  2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def grad_clip(net_list, max_grad_norm):
    for net in net_list:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

def cal_acc(logits, y_true):
    _, ind = torch.max(logits, dim=1)
    acc = torch.sum((ind == y_true).type(torch.FloatTensor))/ y_true.size(1) / y_true.size(0)
    return acc