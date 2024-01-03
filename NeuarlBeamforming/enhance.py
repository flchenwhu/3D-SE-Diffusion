import argparse
import os


import numpy as np
import soundfile as sf
import torch
import torch.utils.data as utils
import librosa
from pathlib import Path


from models.FaSNet import FaSNet_origin, FaSNet_TAC
from models.MMUB import MIMO_UNet_Beamforming
from models.MMULB import MMULB
from utility_functions import load_model, save_model

def enhance_sound(predictors, model, device, length, overlap):
    '''
    Compute enhanced waveform using a trained model,
    applying a sliding crossfading window
    '''

    def pad(x, d):
        #zeropad to desired length
        pad = torch.zeros((x.shape[0], x.shape[1], d))
        pad[:,:,:x.shape[-1]] = x
        return pad

    def xfade(x1, x2, fade_samps, exp=1.):
        #simple linear/exponential crossfade and concatenation
        out = []
        fadein = np.arange(fade_samps) / fade_samps
        fadeout = np.arange(fade_samps, 0, -1) / fade_samps
        fade_in = fadein * exp
        fade_out = fadeout * exp
        x1[:,:,-fade_samps:] = x1[:,:,-fade_samps:] * fadeout
        x2[:,:,:fade_samps] = x2[:,:,:fade_samps] * fadein
        left = x1[:,:,:-fade_samps]
        center = x1[:,:,-fade_samps:] + x2[:,:,:fade_samps]
        end = x2[:,:,fade_samps:]
        return np.concatenate((left,center,end), axis=-1)

    overlap_len = int(length*overlap)  #in samples
    total_len = predictors.shape[-1]
    starts = np.arange(0,total_len, overlap_len)  #points to cut
    #iterate the sliding frames
    for i in range(len(starts)):
        start = starts[i]
        end = starts[i] + length
        if end < total_len:
            cut_x = predictors[:,:,start:end]
        else:
            #zeropad the last frame
            end = total_len
            cut_x = pad(predictors[:,:,start:end], length)

        #compute model's output
        cut_x = cut_x.to(device)
        predicted_x = model(cut_x, device)
        predicted_x = predicted_x.cpu().numpy()

        #reconstruct sound crossfading segments
        if i == 0:
            recon = predicted_x
        else:
            recon = xfade(recon, predicted_x, overlap_len)

    #undo final pad
    recon = recon[:,:,:total_len]

    return recon




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #i/o parameters
    parser.add_argument('--model_path', type=str, default='RESULTS/Task1/032_0.018904.pth')
    parser.add_argument('--results_path', type=str, default='RESULTS/Task1/enahnce')
    parser.add_argument('--save_sounds_freq', type=int, default=400)
    #dataset parameters
    parser.add_argument('--predictors_path', type=str, default='DATASETS/processed100pad_dev/task1_predictors_test_uncut.pkl')
    parser.add_argument('--target_path', type=str, default='DATASETS/processed100pad_dev/task1_target_test_uncut.pkl')
    parser.add_argument('--sr', type=int, default=16000)
    #reconstruction parameters
    parser.add_argument('--segment_length', type=int, default=76672)
    parser.add_argument('--segment_overlap', type=float, default=0.5)
    #model parameters
    parser.add_argument('--architecture', type=str, default='MMULB',
                        help="model name")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--enc_dim', type=int, default=64)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--segment_size', type=int, default=24)
    parser.add_argument('--nspk', type=int, default=1)
    parser.add_argument('--win_len', type=int, default=16)
    parser.add_argument('--context_len', type=int, default=16)
    parser.add_argument('--fft_size', type=int, default=512)
    parser.add_argument('--hop_size', type=int, default=128)
    parser.add_argument('--input_channel', type=int, default=8)

    args = parser.parse_args()



    sound_path = "H:/dataset/2022L3DSE/L3DAS22_Task1_dev/data/84-121123-0000_A.wav"
    samples, sr = librosa.load(sound_path, 16000, mono=False)
    sound_path = Path(sound_path).as_posix()
    B_sound_path = sound_path[:-5] + 'B' +  sound_path[-4:]  #change A with B
    samples_B, sr = librosa.load(B_sound_path, 16000, mono=False)
    samples = np.concatenate((samples, samples_B), axis=-2)




    # samples = pad(samples, size=int(sr_task1 * args.pad_length))
    # samples_target = pad(samples_target, size=int(sr_task1 * args.pad_length))
    # predictors.append(samples)
    # target.append(samples_target)
    x = torch.tensor(samples).float()
    print(x.shape)
    device = 'cuda:0'
    model = MMULB(fft_size=args.fft_size,
                  hop_size=args.hop_size,
                  input_channel=args.input_channel)
    model = model.to(device)
    state = load_model(model, None, args.model_path, args.use_cuda)


    outputs = enhance_sound(x, model, device, args.segment_length, args.segment_overlap)

    outputs = np.squeeze(outputs)
    sounds_dir = os.path.join(args.results_path, 'sounds')
    if not os.path.exists(sounds_dir):
        os.makedirs(sounds_dir)
