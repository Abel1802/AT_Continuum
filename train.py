import numpy as np
import os
from pathlib import Path
from hydra import utils
import argparse
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Encoder, Decoder, PitchClassifier, PitchAE
from data_loader import ContinuumDateset
from utils import get_logger, cal_acc, grad_clip


def load_data(data_dir, batch_size):
    data_set = ContinuumDateset(f"{data_dir}/mels", f"{data_dir}/f0")
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    return data_loader


def load_model(device):
    pitch_ae = PitchAE().to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    classifier = PitchClassifier().to(device)
    return pitch_ae, encoder, decoder, classifier


def main():
    # Hyper-parameters
    weight_at = 0.05
    exp_name = f"F001_mel_f0_disentangle_with_{weight_at}_new"
    saved_dir = f"exp/{exp_name}/checkpoint/"
    os.makedirs(saved_dir, exist_ok=True)
    logger = get_logger(f"{saved_dir}/result.log")
    learning_rate = 1e-4
    betas = (0.5, 0.9)
    data_dir = '/disk2/lz/workspace/data_new/F001'
    batch_size = 16
    beta_dis = 1
    beta_gen = 1
    beta_clf = 1
    pitch_ae_iter = 2000
    enc_pre_iter = 10000
    dis_pre_iter = 10000
    train_iter = 50001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"exp: {exp_name}, start training on {device}...")

    # data
    data_loader = load_data(data_dir, batch_size)

    # model
    pitch_ae, encoder, decoder, classifier = load_model(device)
    pitch_ae_opt = torch.optim.Adam(pitch_ae.parameters(), lr=learning_rate, betas=betas)
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=betas)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=betas)
    clf_opt = torch.optim.Adam(classifier.parameters(), lr=learning_rate, betas=betas)

    '''Step0. Pretrain PitchAE '''
    for iteration in range(pitch_ae_iter):
        x, c = next(iter(data_loader))
        c = c.to(device)
        c_pred = pitch_ae(c)
        rec_loss = pitch_ae.loss_function(c, c_pred)

        pitch_ae_opt.zero_grad()
        rec_loss.backward()
        pitch_ae_opt.step()
        if iteration % 100 ==0:
            logger.info('Iteration: [{:5d}/{:5d}] | \
                  rec_loss = {:.4f}'.format(iteration, pitch_ae_iter, rec_loss))
    torch.save(pitch_ae.state_dict(), f'{saved_dir}/pitch_ae.pkl')

    ''' Step1. Pretrain G (Encoder & Decoder)'''
    for iteration in range(enc_pre_iter):
        x, c = next(iter(data_loader))
        x = x.to(device)
        c = c.to(device)
        emb = pitch_ae.get_f0_emb(c)
        z = encoder(x)
        y = decoder(z, emb)
        rec_loss = F.mse_loss(x, y)
        # backpropogation
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        rec_loss.backward()
        enc_opt.step()
        dec_opt.step()
        if iteration % 100 == 0:
            logger.info("Pretrain G|  [{:5d}/{}] | rec_loss = {:.4f} \
            ".format(iteration, enc_pre_iter, rec_loss))
    torch.save(encoder.state_dict(), f'{saved_dir}/encoder_before.pkl')
    torch.save(decoder.state_dict(), f'{saved_dir}/decoder_before.pkl')

    ''' Step2. Pretrain D (Discriminator)'''
    for iteration in range(dis_pre_iter):
        x, c = next(iter(data_loader))
        x = x.to(device)
        c = c.to(device)
        emb = pitch_ae.get_f0_emb(c)
        _, idx = torch.max(emb, dim=1)
        z = encoder(x)
        p = classifier(z)
        clf_loss = nn.CrossEntropyLoss()(p, idx)
        acc = cal_acc(p, idx)

        # backpropogation
        clf_opt.zero_grad()
        clf_loss.backward()
        clf_opt.step()
        if iteration % 100 == 0:
            logger.info("Pretrain D| [{:5d}/{}] | clf_loss = {:.4f}, \
                    acc = {:.4f}".format(iteration, dis_pre_iter, clf_loss, acc))
    torch.save(classifier.state_dict(), f'{saved_dir}/classifier_before.pkl')


    ''' Step3. Adversarial Training G & D  '''
    for iteration in range(train_iter):
        if iteration < 20000:
            a = weight_at * (iteration/20000)
        else:
            a = weight_at
        # train D
        for i in range(5):
            x, c = next(iter(data_loader))
            x = x.to(device)
            c = c.to(device)
            emb = pitch_ae.get_f0_emb(c)
            _, idx = torch.max(emb, dim=1)
            z = encoder(x)
            p = classifier(z)
            clf_loss = nn.CrossEntropyLoss()(p, idx)
            acc = cal_acc(p, idx)
            clf_opt.zero_grad()
            clf_loss.backward()
            grad_clip([classifier], 5)
            clf_opt.step()
            logger.info('Train D | [{:5d}/{}] | clf_loss = {:.4f} \
                    acc = {:.4f}'.format(iteration, train_iter, clf_loss, acc))

        # train G
        x, c = next(iter(data_loader))
        x = x.to(device)
        c = c.to(device)
        emb = pitch_ae.get_f0_emb(c)
        _, idx = torch.max(emb, dim=1)
        z = encoder(x)
        p = classifier(z)
        y = decoder(z, emb)
        rec_loss = F.mse_loss(x, y)
        clf_loss = nn.CrossEntropyLoss()(p, idx)
        loss = rec_loss - a * clf_loss
        acc = cal_acc(p, idx)

        enc_opt.zero_grad()
        dec_opt.zero_grad()
        loss.backward()
        grad_clip([encoder, decoder], 5)
        enc_opt.step()
        dec_opt.step()
        logger.info('Train G | [{:5d}/{}] | rec_loss = {:.4f} clf_loss = {:.4f} a = {:.2e} \
               acc = {:.4f}'.format(iteration, train_iter, rec_loss, clf_loss, a, acc))

        if iteration % 10000 == 0 and iteration !=0:
            torch.save(encoder.state_dict(), f'{saved_dir}/encoder_after_{iteration}.pkl')
            torch.save(decoder.state_dict(), f'{saved_dir}/decoder_after_{iteration}.pkl')
            torch.save(classifier.state_dict(), f'{saved_dir}/classifier_after_{iteration}.pkl')
            # Plot mel
            plt.figure(figsize=(10, 8))
            plt.subplot(121)
            plt.imshow(x[0].squeeze(0).detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.subplot(122)
            plt.imshow(y[0].squeeze(0).detach().cpu().numpy(), origin='lower', aspect='auto')
            plt.savefig(f'exp/{exp_name}/after_{iteration}_pred_mel.png')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    main()
    
