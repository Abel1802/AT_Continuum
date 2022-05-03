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

from preprocess.tacotron.norm_utils import get_f0, get_spectrograms


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


def process_sublist_wav(sublist):
        '''Processing a batch wavs of a sublist.

        '''
        for wav_path in sublist:
            path_lst = wav_path.split('/')
            name, spk = path_lst[-1], path_lst[-2]
            # Write f0 or f0_target
            f0 = get_f0(wav_path)
            f0_t = f0_quanti(f0)
            np.save(os.path.join('./data/F001/frame_hop_00625/f0_target/', f'{spk}_{name[:-4]}.npy'), f0_t)

            # Write mag_spectrogram
            mel, mag = get_spectrograms(wav_path)
            np.save(os.path.join('./data/F001/frame_hop_00625/mel/', f'{spk}_{name[:-4]}.npy'), mel)

            # Write First-Formant
            fir_formant = get_first_formant(wav_path)
            F1_t = F1_quanti(fir_formant)
            # np.save(os.path.join('./data/F001/frame_hop_00625/first_formants/', f'{spk}_{name[:-4]}.npy'), fir_formant)
            np.save(os.path.join('./data/F001/frame_hop_00625/F1_target/', f'{spk}_{name[:-4]}.npy'), F1_t)


def multi_wavs(wav_dir):
    num_workers = cpu_count()
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    wavs = []
    ''' MonoData '''
    # for spk in os.listdir(wav_dir):
    #     for name in os.listdir(os.path.join(wav_dir, spk)):
    #         wav_path = os.path.join(wav_dir, spk, name)
    #         if wav_path.endswith('.wav'):
    #             wavs.append(wav_path)
    '''Signal person '''
    for name in os.listdir(wav_dir):
        wav_path = os.path.join(wav_dir, name)
        if wav_path.endswith('.wav'):
            wavs.append(wav_path)
    print(f'wavs total: {len(wavs)}')
    
    batch_size = len(wavs)//num_workers 
    wavs_list = []
    for start in range(0, len(wavs), batch_size):
        wavs_list.append(wavs[start:start+batch_size])
    for s in wavs_list:
        futures.append(executor.submit(
            partial(process_sublist_wav, s)))
    for future in tqdm(futures):
        future.result()
    print('Finished!')


def mkdir_f0_t(wav_dir):
    '''Extract f0 from wav.

        Args: wav_dir: contain F001/ F002/ ...
    '''
    spk_lst = os.listdir(wav_dir)
    for spk in tqdm(spk_lst):
        spk_path = os.path.join(wav_dir, spk)
        wav_lst = os.listdir(spk_path)
        for name in wav_lst:
            wav_path = os.path.join(spk_path, name)
            if wav_path.endswith('.wav'):
                f0 = get_f0(wav_path)
                f0_t = f0_quanti(f0)
                np.save(os.path.join('./data/MonoData/f0_target/', f'{spk}_{name[:-4]}.npy'), f0_t)


def get_formant(wav_path, formant_num=1):
    sound = parselmouth.Sound(wav_path)
    formant = call(sound, "To Formant (burg)...", 0.0, 5.0, 5500.0, 0.025, 50.0)
    frames = call(formant, "List all frame times")
    first_formants = []
    for time in frames:
        f1 = call(formant, "Get value at time", formant_num, time, "hertz", "Linear")
        first_formants.append(f1)
    
    f0 = get_f0(wav_path)
    # force the frame numbers of first_formant equal mel
    if len(f0) > len(first_formants):
        padded_len  = len(f0) - len(first_formants)
        first_formants = np.pad(first_formants, (padded_len, 0), 'constant', constant_values=(0, 0))
    else:
        first_formants = first_formants[:len(f0)]
    return first_formants


if __name__ == '__main__':
    wav_path = '../data/F001/'
    multi_wavs(wav_path)

    # wav_path = '../data/F001/'
    # wavList = os.listdir(wav_path)
    # for name in wavList:
    #     wav = os.path.join(wav_path, name)
    #     get_first_formant(wav)
   


