import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


def pad_layer(inp, layer, is_2d=False):
    if type(layer.kernel_size) == tuple:
        kernel_size = layer.kernel_size[0]
    else:
        kernel_size = layer.kernel_size
    if not is_2d:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2)
    else:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1, kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp,
            pad=pad,
            mode='reflect')
    out = layer(inp)
    return out

def pixel_shuffle_1d(inp, upscale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= upscale_factor

    out_width = in_width * upscale_factor
    inp_view = inp.contiguous().view(batch_size, channels, upscale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out

def upsample(x, scale_factor=2):
    x_up = F.upsample(x, scale_factor=2, mode='nearest')
    return x_up

def RNN(inp, layer):
    inp_permuted = inp.permute(0, 2, 1)
    out_permuted, _ = layer(inp_permuted)
    out_rnn = out_permuted.permute(0, 2, 1)
    return out_rnn

def linear(inp, layer):
    batch_size = inp.size(0)
    hidden_dim = inp.size(1)
    seq_len = inp.size(2)
    inp_permuted = inp.permute(0, 2, 1)
    inp_expand = inp_permuted.contiguous().view(batch_size*seq_len, hidden_dim)
    out_expand = layer(inp_expand)
    out_permuted = out_expand.view(batch_size, seq_len, out_expand.size(1))
    out = out_permuted.permute(0, 2, 1)
    return out

def append_emb(emb, expand_size, output):
    emb = emb.unsqueeze(dim=2)
    emb_expand = emb.expand(emb.size(0), emb.size(1), expand_size)
    output = torch.cat([output, emb_expand], dim=1)
    return output


class PitchEncoder(nn.Module):
    def __init__(self, c_in=1, c_h=512, n_class=80, dp=0.1, ns=0.01):
        super(PitchEncoder, self).__init__()
        self.dp, self.ns = dp, ns
        self.conv1 = nn.Conv1d(c_in, c_h, kernel_size=5)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv3 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv5 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv7 = nn.Conv1d(c_h, c_h//2, kernel_size=3)
        self.conv8 = nn.Conv1d(c_h//2, c_h//4, kernel_size=3)
        self.conv9 = nn.Conv1d(c_h//4, n_class, kernel_size=1)
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)

    def conv_block(self, x, conv_layers, after_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        out = self.conv_block(x, [self.conv1, self.conv2], [self.drop1], res=False)
        out = self.conv_block(out, [self.conv7, self.conv8], [self.drop4], res=False)
        out = self.conv9(out)
        # softmax
        out = torch.softmax(out, dim=1)
        return out


class PitchDecoder(nn.Module):
    def __init__(self, c_in=80, c_h=512, n_class=1, dp=0.1, ns=0.01):
        super(PitchDecoder, self).__init__()
        self.dp, self.ns = dp, ns
        self.conv1 = nn.Conv1d(c_in, c_h, kernel_size=5)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv3 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv5 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv7 = nn.Conv1d(c_h, c_h//2, kernel_size=3)
        self.conv8 = nn.Conv1d(c_h//2, c_h//4, kernel_size=3)
        self.conv9 = nn.Conv1d(c_h//4, n_class, kernel_size=1)
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h//4)

    def conv_block(self, x, conv_layers, after_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        out = self.conv_block(x, [self.conv1, self.conv2], [self.drop1], res=False)
        out = self.conv_block(out, [self.conv7, self.conv8], [self.drop4], res=False)
        out = self.conv9(out)
        return out


class PitchAE(nn.Module):
    def __init__(self, c_in=1, c_h=512, n_class=80):
        super(PitchAE, self).__init__()
        self.pitch_encoder = PitchEncoder(c_in=c_in, c_h=c_h, n_class=n_class)
        self.pitch_decoder = PitchDecoder(c_in=n_class, c_h=c_h, n_class=c_in)
    
    def forward(self, f0):
        if len(f0.shape) == 2:
            f0 = f0.unsqueeze(1)
        f0_emb = self.pitch_encoder(f0)
        out = self.pitch_decoder(f0_emb)
        f01_pred = out.squeeze(1)
        return f01_pred

    def get_f0_emb(self, f0):
        if len(f0.shape) == 2:
            f0 = f0.unsqueeze(1)
        f0_emb = self.pitch_encoder(f0)
        return f0_emb

    def loss_function(self, f0, f0_pred, use_l1_loss=False):
        if use_l1_loss:
            loss = F.mse_loss(f0, f0_pred) + F.l1_loss(f0, f0_pred)
        else:
            loss = F.mse_loss(f0, f0_pred)
        return loss



class PatchDiscriminator(nn.Module):
    def __init__(self, n_class=80, ns=0.2, dp=0.1):
        super(PatchDiscriminator, self).__init__()
        self.ns = ns
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=(2,1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=(2,1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=(2,1))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=(2,1))
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=(2,1))
        self.conv6 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=(1,2))
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=(1,2))
        self.conv9 = nn.Conv2d(64, 32, kernel_size=3, stride=(1,2))
        self.conv10 = nn.Conv2d(32, 1, kernel_size=(11, 4))
        #self.conv_classify = nn.Conv2d(512, n_class, kernel_size=(17, 4))
        self.conv_classify = nn.Conv2d(256, n_class, kernel_size=(17, 1))
        self.drop1 = nn.Dropout2d(p=dp)
        self.drop2 = nn.Dropout2d(p=dp)
        self.drop3 = nn.Dropout2d(p=dp)
        self.drop4 = nn.Dropout2d(p=dp)
        self.drop5 = nn.Dropout2d(p=dp)
        self.drop6 = nn.Dropout2d(p=dp)
        self.ins_norm1 = nn.InstanceNorm2d(self.conv1.out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(self.conv2.out_channels)
        self.ins_norm3 = nn.InstanceNorm2d(self.conv3.out_channels)
        self.ins_norm4 = nn.InstanceNorm2d(self.conv4.out_channels)
        self.ins_norm5 = nn.InstanceNorm2d(self.conv5.out_channels)
        self.ins_norm6 = nn.InstanceNorm2d(self.conv6.out_channels)

    def conv_block(self, x, conv_layer, after_layers):
        out = pad_layer(x, conv_layer, is_2d=True)
        out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        return out 

    def forward(self, x, classify=False):
        x = torch.unsqueeze(x, dim=1)
        out = self.conv_block(x, self.conv1, [self.ins_norm1, self.drop1])
        # print(f'after conv1: {out.shape}')
        out = self.conv_block(out, self.conv2, [self.ins_norm2, self.drop2])
        out = self.conv_block(out, self.conv3, [self.ins_norm3, self.drop3])
        out = self.conv_block(out, self.conv4, [self.ins_norm4, self.drop4])
        out = self.conv_block(out, self.conv5, [self.ins_norm5, self.drop5])
        out = self.conv_block(out, self.conv6, [self.ins_norm6, self.drop6])
        # print(f'after conv6: {out.shape}')
        # GAN output value
        val = self.conv7(out)
        val = self.conv8(val)
        val = self.conv9(val)
        # print(f'after conv9: {val.shape}')
        val = self.conv10(val)
        val = val.view(val.size(0), -1)
        mean_val = torch.mean(val, dim=1)
        if classify:
            # classify
            logits = self.conv_classify(out)
            # logits = logits.view(logits.size(0), -1)
            logits = logits.squeeze(2)
            return mean_val, logits
        else:
            return mean_val

class PitchClassifier(nn.Module):
    def __init__(self, c_in=512, c_h=512, n_class=80, dp=0.1, ns=0.01):
        super(PitchClassifier, self).__init__()
        self.dp, self.ns = dp, ns
        self.conv1 = nn.Conv1d(c_in, c_h, kernel_size=5)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv3 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv5 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv7 = nn.Conv1d(c_h, c_h//2, kernel_size=3)
        self.conv8 = nn.Conv1d(c_h//2, c_h//2, kernel_size=3)
        self.conv9 = nn.Conv1d(c_h//2, n_class, kernel_size=1)
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h//2)

    def conv_block(self, x, conv_layers, after_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        out = self.conv_block(x, [self.conv1, self.conv2], [self.ins_norm1, self.drop1], res=False)
        out = self.conv_block(out, [self.conv3, self.conv4], [self.ins_norm2, self.drop2], res=True)
        out = self.conv_block(out, [self.conv5, self.conv6], [self.ins_norm3, self.drop3], res=True)
        out = self.conv_block(out, [self.conv7, self.conv8], [self.ins_norm4, self.drop4], res=False)
        out = self.conv9(out)
        #out = out.view(out.size()[0], -1)
        #print(f"after classifier {out.shape}")
        return out

class Decoder(nn.Module):
    def __init__(self, c_in=512, c_out=80, c_h=512, c_a=80, emb_size=128, ns=0.2):
        super(Decoder, self).__init__()
        self.ns = ns
        self.conv1 = nn.Conv1d(c_in, c_h, kernel_size=3)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.conv3 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.conv5 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.dense1 = nn.Linear(c_h, c_h)
        self.dense2 = nn.Linear(c_h, c_h)
        self.dense3 = nn.Linear(c_h, c_h)
        self.dense4 = nn.Linear(c_h, c_h)
        self.RNN = nn.LSTM(input_size=c_h, hidden_size=c_h//2, num_layers=1, bidirectional=True)
        self.dense5 = nn.Linear(2*c_h, c_h)
        self.linear = nn.Linear(c_h, c_out)
        # normalization layer
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h)
        self.ins_norm5 = nn.InstanceNorm1d(c_h)
        # embedding layer
        self.emb1 = nn.Embedding(c_a, c_h)
        self.emb2 = nn.Embedding(c_a, c_h)
        self.emb3 = nn.Embedding(c_a, c_h)
        self.emb4 = nn.Embedding(c_a, c_h)
        self.emb5 = nn.Embedding(c_a, c_h)

        # mapping 80 dimension f0 to 512
        self.f0_conv1 = nn.Conv1d(c_a, c_in, kernel_size=5, padding=2)
        self.f0_conv2 = nn.Conv1d(c_a, c_in, kernel_size=11, padding=5)
        self.f0_conv3 = nn.Conv1d(c_a, c_in, kernel_size=15, padding=7)

    def conv_block(self, x, conv_layers, norm_layer, emb, res=True):
        # first layer
        x_add = x + emb
        # print(f'before conv1: {x_add.shape}')
        out = pad_layer(x_add, conv_layers[0])
        # print(f'after conv1: {out.shape}')
        # out = x_add
        out = F.leaky_relu(out, negative_slope=self.ns)
        # # upsample by pixelshuffle
        out = pixel_shuffle_1d(out, upscale_factor=1)
        out = out + emb
        out = pad_layer(out, conv_layers[1])
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = norm_layer(out)
        if res:
            # x_up = upsample(x, scale_factor=1)
            out = out + x
        return out

    def dense_block(self, x, emb, layers, norm_layer, res=True):
        out = x
        for layer in layers:
            out = out + emb
            out = linear(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        out = norm_layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x, c):
        # First convert f0'80 to f0'512
        c1 = self.f0_conv1(c)
        c2 = self.f0_conv2(c)
        c3 = self.f0_conv3(c)
        c = c1 + c2 + c3
        # conv layer
        out = self.conv_block(x, [self.conv1, self.conv2], self.ins_norm1, c, res=True )
        out = self.conv_block(out, [self.conv3, self.conv4], self.ins_norm2, c, res=True)
        out = self.conv_block(out, [self.conv5, self.conv6], self.ins_norm3, c, res=True)
        # dense layer
        # out = x
        out = self.dense_block(out, c, [self.dense1, self.dense2], self.ins_norm4, res=True)
        out = self.dense_block(out, c, [self.dense3, self.dense4], self.ins_norm5, res=True)
        out_add = out + c
        # rnn layer
        out_rnn = RNN(out_add, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        #out = append_emb(self.emb5(c), out.size(2), out)
        out = linear(out, self.dense5)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = linear(out, self.linear)
        return out

class Encoder(nn.Module):
    def __init__(self, c_in=80, c_h1=128, c_h2=512, c_h3=128, ns=0.2, dp=0.5):
        super(Encoder, self).__init__()
        self.ns = ns
        self.conv1s = nn.ModuleList(
                [nn.Conv1d(c_in, c_h1, kernel_size=k) for k in range(1, 8)]
            )
        self.conv2 = nn.Conv1d(len(self.conv1s)*c_h1 + c_in, c_h2, kernel_size=1)
        self.conv3 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=1)
        self.conv5 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=1)
        self.conv7 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv8 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=1)
        self.dense1 = nn.Linear(c_h2, c_h2)
        self.dense2 = nn.Linear(c_h2, c_h2)
        self.dense3 = nn.Linear(c_h2, c_h2)
        self.dense4 = nn.Linear(c_h2, c_h2)
        self.RNN = nn.LSTM(input_size=c_h2, hidden_size=c_h3, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(c_h2 + 2*c_h3, c_h2)
        # normalization layer
        self.ins_norm1 = nn.InstanceNorm1d(c_h2)
        self.ins_norm2 = nn.InstanceNorm1d(c_h2)
        self.ins_norm3 = nn.InstanceNorm1d(c_h2)
        self.ins_norm4 = nn.InstanceNorm1d(c_h2)
        self.ins_norm5 = nn.InstanceNorm1d(c_h2)
        self.ins_norm6 = nn.InstanceNorm1d(c_h2)
        # dropout layer
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)
        self.drop5 = nn.Dropout(p=dp)
        self.drop6 = nn.Dropout(p=dp)

    def conv_block(self, x, conv_layers, norm_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in norm_layers:
            out = layer(out)
        if res:
            out = x + out
        return out

    def dense_block(self, x, layers, norm_layers, res=True):
        out = x
        for layer in layers:
            out = linear(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in norm_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        outs = []
        for l in self.conv1s:
            out = pad_layer(x, l)
            outs.append(out)
        out = torch.cat(outs + [x], dim=1)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = self.conv_block(out, [self.conv2], [self.ins_norm1, self.drop1], res=False)
        out = self.conv_block(out, [self.conv3, self.conv4], [self.ins_norm2, self.drop2], res=True)
        out = self.conv_block(out, [self.conv5, self.conv6], [self.ins_norm3, self.drop3], res=True)
        out = self.conv_block(out, [self.conv7, self.conv8], [self.ins_norm4, self.drop4], res=True)
        # dense layer
        out = self.dense_block(out, [self.dense1, self.dense2], [self.ins_norm5, self.drop5], res=True)
        out = self.dense_block(out, [self.dense3, self.dense4], [self.ins_norm6, self.drop6], res=True)
        out_rnn = RNN(out, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = linear(out, self.linear)
        out = F.leaky_relu(out, negative_slope=self.ns)
        #print(f"after encoder {out.shape}")
        return out



if __name__ == "__main__":
    mels = torch.randn(256, 80, 128)
    f0s = torch.randn(256, 128)

    pitch_ae = PitchAE()
    encoder = Encoder()
    decoder = Decoder()
    pitch_classifier = PitchClassifier()

    f0s_emb = pitch_ae.get_f0_emb(f0s)
    print(f"f0s_emb: {f0s_emb.shape}")

    z = encoder(mels)
    print(f"z: {z.shape}")

    logits = pitch_classifier(z)
    print(f"logits: {logits.shape}")

    pred = decoder(z, f0s_emb)
    print(f"pred: {pred.shape}")


    
