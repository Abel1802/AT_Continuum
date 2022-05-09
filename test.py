import torch
import numpy as np
import librosa
import os
import soundfile as sf
import pyworld as pw
import matplotlib.pyplot as plt 
import kaldiio
import subprocess

from model import PitchAE
from model import Encoder
from model import Decoder
from spectrogram import logmelspectrogram
from preprocess.tacotron.norm_utils import get_spectrograms
from preprocess.tacotron.norm_utils import get_f0


def main():
    exp_name = 'VC_F001_mel_f0_disentangle_with_0.01'
    device = torch.device('cuda:0')

    # Prepare test data
    src_wav_path = "../bei1.wav"
    ref_wav_path = "../bei4_fix.wav"
    out_dir = "converted"
    os.makedirs(out_dir, exist_ok=True)

    # Load saved model
    pitch_ae = PitchAE().to(device)
    encoder_before = Encoder().to(device)
    decoder_before = Decoder().to(device)
    encoder_after = Encoder().to(device)
    decoder_after = Decoder().to(device)

    pitch_ae.load_state_dict(torch.load(f"exp/{exp_name}/pitch_ae.pkl"))
    encoder_before.load_state_dict(torch.load(f"exp/{exp_name}/encoder_before.pkl"))
    decoder_before.load_state_dict(torch.load(f"exp/{exp_name}/decoder_before.pkl"))
    encoder_after.load_state_dict(torch.load(f"exp/{exp_name}/encoder_after.pkl"))
    decoder_after.load_state_dict(torch.load(f"exp/{exp_name}/decoder_after.pkl"))

    pitch_ae.eval()
    encoder_before.eval()
    decoder_before.eval()
    encoder_after.eval()
    decoder_after.eval()

    # Run inference
    src_mel, _ = get_spectrograms(src_wav_path)
    ref_mel, _ = get_spectrograms(ref_wav_path)
    src_lf0 = get_f0(src_wav_path)
    ref_lf0 = get_f0(ref_wav_path)
    src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(device)
    src_lf0 = torch.FloatTensor(src_lf0).unsqueeze(0).to(device)
    ref_lf0 = torch.FloatTensor(ref_lf0).unsqueeze(0).to(device)
    ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
    with torch.no_grad():
        lf01_pred = pitch_ae(src_lf0)
        lf02_pred = pitch_ae(ref_lf0)

        lf01_embs = pitch_ae.get_f0_emb(src_lf0)
        lf02_embs = pitch_ae.get_f0_emb(ref_lf0)
        z1 = encoder_before(src_mel)
        z2 = encoder_before(ref_mel)
        output1 = decoder_before(z1, lf01_embs)
        output1_2 = decoder_before(z1, lf02_embs)
        output2_1 = decoder_before(z2, lf01_embs)
        output2 = decoder_before(z2, lf02_embs)

        # Plot mel
        plt.figure(figsize=(20, 8))
        plt.subplot(161)
        plt.imshow(src_mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(162)
        plt.imshow(output1.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(163)
        plt.imshow(output1_2.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(164)
        plt.imshow(output2_1.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(165)
        plt.imshow(output2.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(166)
        plt.imshow(ref_mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.savefig(f'exp/{exp_name}/before_bei_{exp_name}.png')

        z1_after = encoder_after(src_mel)
        z2_after = encoder_after(ref_mel)
        output1_after = decoder_after(z1_after, lf01_embs)
        output1_2_after = decoder_after(z1_after, lf02_embs)
        output2_1_after = decoder_after(z2_after, lf01_embs)
        output2_after = decoder_after(z2_after, lf02_embs)

        # Plot mel
        plt.figure(figsize=(20, 8))
        plt.subplot(161)
        plt.imshow(src_mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(162)
        plt.imshow(output1_after.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(163)
        plt.imshow(output1_2_after.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(164)
        plt.imshow(output2_1_after.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(165)
        plt.imshow(output2_after.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(166)
        plt.imshow(ref_mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.savefig(f'exp/{exp_name}/after_bei_{exp_name}.png')

        # Plot lf0
        plt.figure(figsize=(20, 8))
        plt.plot(src_lf0.squeeze(0).cpu().numpy(), label='src')
        plt.plot(lf01_pred.squeeze(0).cpu().numpy(), label='src_pred')
        plt.plot(ref_lf0.squeeze(0).cpu().numpy(), label='ref')
        plt.plot(lf02_pred.squeeze(0).cpu().numpy(), label='ref_pred')
        plt.legend()
        plt.savefig(f'exp/{exp_name}/bei_{exp_name}_lf0.png')

        



if __name__ == '__main__':
    main()