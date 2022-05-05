import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
import parselmouth
from parselmouth.praat import call

class ContinuumDateset(Dataset):
    '''A Dataset contain f0 and spectrogram.

        Args:
            f0_dir, the path which contain f0 npy-file.
            spec_dir, the path which contain lin-spec npy-file.
    '''
    def __init__(self, spec_dir, f0_dir):
        self.n_frames = 64
        self.f0_dir = f0_dir
        self.spec_dir = spec_dir
        self.f0_lst = os.listdir(f0_dir)
        self.spec_lst = os.listdir(spec_dir)

    def get_f0_spec(self, f0_path, spec_path):
        f0 = np.load(f0_path)
        f0 = f0/600
        mel = np.load(spec_path).T
        melt = mel
        f0t = f0
        while mel.shape[-1] <= self.n_frames:
            mel = np.concatenate([mel, melt], -1)
            f0 = np.concatenate([f0, f0t], 0)
        zero_idxs = np.where(f0 == 0.0)[0]
        nonzero_idxs = np.where(f0 != 0.0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(f0[nonzero_idxs])
            std = np.std(f0[nonzero_idxs])
            if std == 0:
                f0 -= mean
                f0[zero_idxs] = 0.0
            else:
                f0 = (f0 - mean) / (std + 1e-8)
                f0[zero_idxs] = 0.0
        # print(mel.shape)
        pos = np.random.randint(0, mel.shape[-1] - self.n_frames)
        mel = mel[:, pos:pos+self.n_frames]
        f0 = f0[pos:pos+self.n_frames]
        return torch.from_numpy(mel).float(), torch.from_numpy(f0).float()

    def __getitem__(self, index):
        f0_path = os.path.join(self.f0_dir, self.f0_lst[index])
        spec_path = os.path.join(self.spec_dir, self.spec_lst[index])
        spec, f0 = self.get_f0_spec(f0_path, spec_path)
        return spec, f0

    def __len__(self):
        return len(self.spec_lst)