import torch
import numpy as np
import librosa
import resampy
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
from utils import get_formant


def extract_logmel(wav_path, mean, std, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    wav, _ = librosa.effects.trim(wav, top_db=60)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    # duration = len(wav)/fs
    assert fs == 16000
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    mel = logmelspectrogram(
                x=wav,
                fs=fs,
                n_mels=80,
                n_fft=400,
                n_shift=160,
                win_length=400,
                window='hann',
                fmin=80,
                fmax=7600,
            )
    mel = (mel - mean) / (std + 1e-8)
    tlen = mel.shape[0]
    frame_period = 160 / fs * 1000
    f0, timeaxis = pw.dio(wav.astype("float64"), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype("float64"), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype("float32")
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(
        f0[nonzeros_indices]
    )  # for f0(Hz), lf0 > 0 when f0 != 0
    mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
    return mel, f0/600


def main():
    formant = False
    resume_iter = 20000
    exp_name = 'F001_mel_f0_disentangle_with_0.01'
    device = torch.device('cuda:0')
    out_dir = f"exp/{exp_name}/converted/"
    os.makedirs(out_dir, exist_ok=True)
    mel_stats = np.load("/disk2/lz/workspace/data_new/F001/mel_stats.npy")
    mel_mean, mel_std = mel_stats[0], mel_stats[1]

    # test_data
    # feat_writer = kaldiio.WriteHelper(
            # "ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir) + "/feats.1))

    feat_writer = kaldiio.WriteHelper(
            "ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir) + "/feats.1")
        )

    # Prepare test data
    src_wav_path = "../data_new/test_wavs/bei1.wav"
    ref_wav_path = "../data_new/test_wavs/bei4_fix.wav"
    out_filename = os.path.basename(src_wav_path).split(".")[0]

    # Load saved model
    pitch_ae = PitchAE().to(device)
    encoder_before = Encoder().to(device)
    decoder_before = Decoder().to(device)
    encoder_after = Encoder().to(device)
    decoder_after = Decoder().to(device)

    pitch_ae.load_state_dict(torch.load(f"exp/{exp_name}/checkpoint/pitch_ae.pkl"))
    encoder_before.load_state_dict(torch.load(f"exp/{exp_name}/checkpoint/encoder_before.pkl"))
    decoder_before.load_state_dict(torch.load(f"exp/{exp_name}/checkpoint/decoder_before.pkl"))
    encoder_after.load_state_dict(torch.load(f"exp/{exp_name}/checkpoint/encoder_after_{resume_iter}.pkl"))
    decoder_after.load_state_dict(torch.load(f"exp/{exp_name}/checkpoint/decoder_after_{resume_iter}.pkl"))

    pitch_ae.eval()
    encoder_before.eval()
    decoder_before.eval()
    encoder_after.eval()
    decoder_after.eval()

    # Run inference
    src_mel, src_lf0 = extract_logmel(src_wav_path, mel_mean, mel_std)
    ref_mel, ref_lf0 = extract_logmel(ref_wav_path, mel_mean, mel_std)
    # get formant
    if formant:
        src_lf0 = get_formant(src_wav_path, src_lf0, formant_num=2)
        src_lf0 = np.array(src_lf0) / 5500
        ref_lf0 = get_formant(ref_wav_path, ref_lf0, formant_num=2)
        ref_lf0 = np.array(ref_lf0) / 5500

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

        feat_writer[out_filename + "_converted_before"] = output1_2.squeeze(0).cpu().numpy().T
        feat_writer[out_filename + "_source"] = src_mel.squeeze(0).cpu().numpy().T
        feat_writer[out_filename + "_reference"] = ref_mel.squeeze(0).cpu().numpy().T

        z1_after = encoder_after(src_mel)
        z2_after = encoder_after(ref_mel)
        output1_after = decoder_after(z1_after, lf01_embs)
        output1_2_after = decoder_after(z1_after, lf02_embs)
        output2_1_after = decoder_after(z2_after, lf01_embs)
        output2_after = decoder_after(z2_after, lf02_embs)

        # Continuum
        b_ints = np.linspace(0, 1, 10)
        lf0_ints = [lf01_embs * (1-b) + lf02_embs * b for b in b_ints]
        mel_ints = [decoder_after(z1_after, lf0_int) for lf0_int in lf0_ints]
        for i in range(10):
            feat_writer[out_filename + f"_{resume_iter}_continuum_{i}"] = mel_ints[i].squeeze(0).cpu().numpy().T

        diff = (src_lf0 - ref_lf0) / 6
        b_ints = np.linspace(2, -8, 11)
        lf0_ints = [lf01_embs + diff * b for b in b_ints]
        mel_ints = [decoder_after(z1_after, lf0_int) for lf0_int in lf0_ints]
        for i in range(11):
            feat_writer[out_filename + f"std_continuum_{i}"] = mel_ints[i].squeeze(0).cpu().numpy().T


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

        feat_writer[out_filename + "_converted_after"] = output1_2_after.squeeze(0).cpu().numpy().T

        # Plot lf0
        plt.figure(figsize=(20, 8))
        plt.plot(src_lf0.squeeze(0).cpu().numpy(), label='src')
        plt.plot(lf01_pred.squeeze(0).cpu().numpy(), label='src_pred')
        plt.plot(ref_lf0.squeeze(0).cpu().numpy(), label='ref')
        plt.plot(lf02_pred.squeeze(0).cpu().numpy(), label='ref_pred')
        plt.legend()
        plt.savefig(f'exp/{exp_name}/bei_{exp_name}_lf0.png')
        
        feat_writer.close()

    # vocoder
    print("synthesize waveform...")
    cmd = [
        "parallel-wavegan-decode",
        "--checkpoint",
        "./vocoder/checkpoint-3000000steps.pkl",
        "--feats-scp",
        f"{str(out_dir)}/feats.1.scp",
        "--outdir",
        str(out_dir),
    ]
    subprocess.call(cmd)







if __name__ == '__main__':
    main()
