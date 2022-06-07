import logging
import torch
import numpy as np
import parselmouth
from parselmouth.praat import call


def get_formant(wav_path, f0, formant_num=1):
    sound = parselmouth.Sound(wav_path)
    formant = call(sound, "To Formant (burg)...", 0.0, 5.0, 5500.0, 0.025, 50.0)
    frames = call(formant, "List all frame times")
    num_formants = []
    for time in frames:
        tmp = call(formant, "Get value at time", formant_num, time, "hertz", "Linear")
        num_formants.append(tmp)

    # force the frame numbers of num_formants equal mel
    if len(f0) > len(num_formants):
        padded_len  = len(f0) - len(num_formants)
        num_formants = np.pad(num_formants, (padded_len, 0), 'constant', constant_values=(0, 0))
    else:
        num_formants = num_formants[:len(f0)]
    return num_formants


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
