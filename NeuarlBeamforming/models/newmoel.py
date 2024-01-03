import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from show import show_params, show_model


class MSWB(nn.Module):
    def __init__(self,
                fft_size=512,
                hop_size=128,
                input_channel=8, # the channel number of input audio
                input_channel2=2, # the channel number of input audio2
                unet_channel=[32,32,32,64,64,96,96,96,128,256],
                kernel_size=[(7,1),(1,7),(8,6),(7,6),(6,5),(5,5),(6,3),(5,3),(6,3),(5,3)],
                stride=[(1,1),(1,1),(2,2),(1,1),(2,2),(1,1),(2,2),(1,1),(2,1),(1,1)],
                ):
        super(MMULB, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = fft_size
        self.valid_freq = int(self.fft_size / 2)

        layer_number = len(unet_channel)
        kernel_number = len(kernel_size)
        stride_number = len(stride)
        assert layer_number==kernel_number==stride_number

        self.kernel = kernel_size
        self.stride = stride


        # encoder setting
        self.encoder = nn.ModuleList()
        self.encoder_channel = [input_channel] + unet_channel
        self.encoder2 = nn.ModuleList()
        self.encoder_channel2 = [input_channel2] + unet_channel

        # decoder setting
        self.decoder = nn.ModuleList()
        self.decoder_outchannel = unet_channel
        self.decoder_inchannel = list(map(lambda x:x[0] + x[1] ,zip(unet_channel[1:] + [0], unet_channel)))
        self.decoder2 = nn.ModuleList()
        self.decoder_outchannel2 = unet_channel
        self.decoder_inchannel2 = list(map(lambda x:x[0] + x[1] ,zip(unet_channel[1:] + [0], unet_channel)))

        self.conv2d = nn.Conv2d(self.decoder_outchannel[0], input_channel, 1, 1)
        self.conv2d2 = nn.Conv2d(self.decoder_outchannel[0], input_channel2, 1, 1)
        self.linear = nn.Linear(self.valid_freq * 2, self.valid_freq * 2)

        for idx in range(layer_number):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.encoder_channel[idx],
                        self.encoder_channel[idx+1],
                        self.kernel[idx],
                        self.stride[idx],
                    ),
                    nn.BatchNorm2d(self.encoder_channel[idx+1]),
                    # nn.LeakyReLU(0.3)
                    nn.LeakyReLU(0.3)
                )
            )

        for idx in range(layer_number):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.decoder_inchannel[-1-idx],
                        self.decoder_outchannel[-1-idx],
                        self.kernel[-1-idx],
                        self.stride[-1-idx]
                    ),
                    nn.BatchNorm2d(self.decoder_outchannel[-1-idx]),
                    # nn.LeakyReLU(0.3)
                    nn.LeakyReLU(0.3)
                )
            )

        for idx in range(layer_number):
            self.encoder2.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.encoder_channel2[idx],
                        self.encoder_channel2[idx+1],
                        self.kernel[idx],
                        self.stride[idx],
                    ),
                    nn.BatchNorm2d(self.encoder_channel2[idx+1]),
                    # nn.LeakyReLU(0.3)
                    nn.LeakyReLU(0.3)
                )
            )

        for idx in range(layer_number):
            self.decoder2.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.decoder_inchannel2[-1-idx],
                        self.decoder_outchannel2[-1-idx],
                        self.kernel[-1-idx],
                        self.stride[-1-idx]
                    ),
                    nn.BatchNorm2d(self.decoder_outchannel2[-1-idx]),
                    # nn.LeakyReLU(0.3)
                    nn.LeakyReLU(0.3)
                )
            )




        # show_model(self)
        # show_params(self)
    def extract_features(self, inputs, device):
        # shape: [B, C, S]
        batch_size, channel, samples = inputs.size()

        features = []
        for idx in range(batch_size):
            # shape: [C, F, T, 2]
            features_batch = torch.stft(
                                inputs[idx, ...],
                                self.fft_size,
                                self.hop_size,
                                self.win_size,
                                torch.hann_window(self.win_size).to(device),
                                pad_mode='constant',
                                onesided=True,
                                return_complex=False)
            features.append(features_batch)
            # print(features_batch)
            # print(features)
            # print(features_batch.shape) #[8, 257, 600, 2]

        # shape: [B, C, F, T, 2]
        features = torch.stack(features, 0)
        features = features[:,:,:self.valid_freq,:,:]
        real_features = features[..., 0]
        imag_features = features[..., 1]

        return real_features, imag_features

    def encode_padding_size(self, kernel_size):
        k_f, k_t = kernel_size
        p_f_s = int(k_f / 2)
        p_t_s = int(k_t / 2)

        p_t_0, p_t_1, p_f_0, p_f_1 = (p_t_s, p_t_s, p_f_s, p_f_s)

        if k_t % 2 == 0:
            p_t_0 = p_t_0 - 1

        if k_f % 2 == 0:
            p_f_0 = p_f_0 - 1

        return (p_t_0, p_t_1, p_f_0, p_f_1)

    def decode_padding_size(self, in_size, target_size):
        i_f, i_t = in_size
        t_f, t_t = target_size
        # print((i_f, i_t),(t_f, t_t))
        p_t_s = int(abs(t_t - i_t) / 2)
        p_f_s = int(abs(t_f - i_f) / 2)

        p_t_0, p_t_1, p_f_0, p_f_1 = (p_t_s, p_t_s, p_f_s, p_f_s)

        if abs(t_t - i_t) % 2 == 1:
            p_t_1 = p_t_1 + 1

        if abs(t_f - i_f) % 2 == 1:
            p_f_1 = p_f_1 + 1

        return (p_t_0, p_t_1, p_f_0, p_f_1)

    def encode_padding_same(self, features, kernel_size):
        p_t_0, p_t_1, p_f_0, p_f_1 = self.encode_padding_size(kernel_size)
        # print((p_t_0, p_t_1, p_f_0, p_f_1))

        features = F.pad(features, (p_t_0, p_t_1, p_f_0, p_f_1))

        return features

    def decode_padding_same(self, features, encoder_features, stride):
        # shape: [B, C, F, T]
        _, _, f, t = features.size()
        _, _, ef, et = encoder_features.size()

        # shape: [F, T]
        sf, st = stride
        tf, tt = (int(ef * sf), int(et * st))

        p_t_0, p_t_1, p_f_0, p_f_1 = self.decode_padding_size((f, t), (tf, tt))
        # print((p_t_0, p_t_1, p_f_0, p_f_1))

        # shape: [B, C, F, T]
        if (p_t_0 != 0) or (p_t_1 != 0):
            features = features[:, :, :, p_t_0:-p_t_1]
        if (p_f_0 != 0) or (p_f_1 != 0):
            features = features[:, :, p_f_0:-p_f_1, :]

        return features

    def forward(self, inputs, device):
        # shape: [B, C, F, T]
        real_features, imag_features = self.extract_features(inputs, device)

        real_features1 = real_features[:, 0, :, :]
        imag_features1 = imag_features[:, 0, :, :]

        # shape: [B, C, F*2, T]
        features = torch.cat((real_features, imag_features), 2)
        features1 = features[:, 0, :, :]
        features1 = torch.unsqueeze(features1, 1) # [3, 1, 512, 600]




        out = features
        # print(out.size())#[3, 8, 512, 600]
        encoder_out = []
        for idx, layer in enumerate(self.encoder):
            # print(out.shape)
            out = self.encode_padding_same(out, self.kernel[idx])
            # print(out.shape)
            out = layer(out)
            # print(out.shape)
            encoder_out.append(out)



        out = encoder_out[-1]
        for idx, layer in enumerate(self.decoder):
            if idx != 0:
                out = torch.cat((out, encoder_out[-1-idx]), 1)
            # print(out.shape)
            out = layer(out)
            # print(out.shape)
            out = self.decode_padding_same(out, encoder_out[-1-idx], self.stride[-1-idx])
            # print(out.shape)
            # to [L, B, C, D]


        # out = out.permute(1, 2, 3, 0)
        out = self.conv2d(out)
        # shape: [B, C, T, F*2]
        out = out.permute(0,1,3,2)
        out = self.linear(out)
        # shape: [B, C, F*2, T]
        out = out.permute(0,1,3,2)

        real_mask = out[:,:,:self.valid_freq,:]
        imag_mask = out[:,:,self.valid_freq:,:]

        est_speech_real = torch.mul(real_features, real_mask) - torch.mul(imag_features, imag_mask)
        est_speech_imag = torch.mul(real_features, imag_mask) + torch.mul(imag_features, real_mask)
        est_speech_stft = torch.complex(est_speech_real, est_speech_imag)   #[3, 8, 256, 600]


        # shape: [B, C, F, T]
        est_speech_stft = torch.sum(est_speech_stft, 1)
        batch_size, frequency, frame = est_speech_stft.size() #[3, 256, 600]
        est_speech_stft = torch.cat((est_speech_stft, torch.zeros(batch_size, 1, frame).to(device)), 1) #[3, 257, 600]


        # shape: [B, S]        #[3, 76672]
        est_speech = torch.istft(
                        est_speech_stft,
                        self.fft_size,
                        self.hop_size,
                        self.win_size,
                        torch.hann_window(self.win_size).to(device))


        est_speech1 = torch.unsqueeze(est_speech, 1)
        real_features2, imag_features2 = self.extract_features(est_speech1, device)
        real_features1 = torch.unsqueeze(real_features1, 1)
        imag_features1 = torch.unsqueeze(imag_features1, 1)
        # print(real_features1.shape,real_features2.shape)#torch.Size([3, 1, 256, 600]) torch.Size([3, 1, 256, 600])
        real_features2 = torch.cat((real_features1, real_features2), 1)
        imag_features2 = torch.cat((imag_features1,imag_features2), 1)

        # shape: [B, C, F*2, T]
        features2 = torch.cat((real_features2, imag_features2), 2)
        out = features2
        # print(out.shape)#torch.Size([3, 2, 512, 600])
        encoder_out = []
        for idx, layer in enumerate(self.encoder2):
            # print(out.shape)
            out = self.encode_padding_same(out, self.kernel[idx])
            # print(out.shape)
            out = layer(out)
            # print(out.shape)
            encoder_out.append(out)


        out = encoder_out[-1]
        for idx, layer in enumerate(self.decoder2):
            if idx != 0:
                out = torch.cat((out, encoder_out[-1 - idx]), 1)
            # print(out.shape)
            out = layer(out)
            # print(out.shape)
            out = self.decode_padding_same(out, encoder_out[-1 - idx], self.stride[-1 - idx])
            # print(out.shape)
            # to [L, B, C, D]

        # print(out.shape) #[3, 32, 512, 600]
        # out = out.permute(3, 0, 1, 2)
        out = self.conv2d2(out)
        # shape: [B, C, T, F*2]
        out = out.permute(0, 1, 3, 2)
        out = self.linear(out)
        # shape: [B, C, F*2, T]
        out = out.permute(0, 1, 3, 2)

        real_mask2 = out[:, :, :self.valid_freq, :]
        imag_mask2 = out[:, :, self.valid_freq:, :]

        est_speech_real = torch.mul(real_features2, real_mask2) - torch.mul(imag_features2, imag_mask2)
        est_speech_imag = torch.mul(real_features2, imag_mask2) + torch.mul(imag_features2, real_mask2)
        est_speech_stft = torch.complex(est_speech_real, est_speech_imag)  # [3, 8, 256, 600]

        # shape: [B, C, F, T]
        est_speech_stft = torch.sum(est_speech_stft, 1)
        batch_size, frequency, frame = est_speech_stft.size()  # [3, 256, 600]
        est_speech_stft = torch.cat((est_speech_stft, torch.zeros(batch_size, 1, frame).to(device)), 1)  # [3, 257, 600]

        # shape: [B, S]        #[3, 76672]
        est_speech = torch.istft(
            est_speech_stft,
            self.fft_size,
            self.hop_size,
            self.win_size,
            torch.hann_window(self.win_size).to(device))

        est_speech = torch.unsqueeze(est_speech, 1)

        # [3, 2, 512, 600]
        # shape: [B, 1, S]
        return est_speech


if __name__ == '__main__':
    '''
    The frame number input to the model must be a multiple of 8, here it's 600.
    Because the torch.stft pads 4 extra frames based on our configurations, the
    frame number of the signal is 596 actually, ie. the duration of the signal
    is 4.792 seconds(76672 sample), while the fft_size is 512, hop_size is 128,
    and the sample_rate is 16000.
    The frequency bin input to the model must be a multiple of 16, here it's 256.
    '''
    frames_num = 240
    fft_size = 512
    hop_size = 128
    batch_size = 3
    audio_channel = 8
    length = int((frames_num - 1) * hop_size + fft_size - 4 * hop_size) # 1.912 seconds
    inputs = torch.rand(batch_size,audio_channel,length)

    model = MMULB()
    out = model(inputs, 'cpu')
    print('input size:', inputs.size())
    print('out size:', out.size())

    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))

