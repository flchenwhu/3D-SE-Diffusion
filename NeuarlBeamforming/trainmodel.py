import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as utils
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.FaSNet import FaSNet_origin, FaSNet_TAC
from models.MMUB import MIMO_UNet_Beamforming
# from models.MMULB import MMULB
from models.newmoel import MSWB
from utility_functions import load_model, save_model

'''
Train our baseline model for the Task1 of the L3DAS22 challenge.
This script saves the model checkpoint, as well as a dict containing
the results (loss and history). To evaluate the performance of the trained model
according to the challenge metrics, please use evaluate_baseline_task1.py.
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''

import argparse
import os
import pickle
import random
import sys

import librosa
import numpy as np

import utility_functions as uf
from pathlib import Path
'''
Process the unzipped dataset folders and output numpy matrices (.pkl files)
containing the pre-processed data for task1 and task2, separately.
Separate training, validation and test matrices are saved.
Command line inputs define which task to process and its parameters.
'''

sound_classes_dict_task2 = {'Chink_and_clink':0,
                           'Computer_keyboard':1,
                           'Cupboard_open_or_close':2,
                           'Drawer_open_or_close':3,
                           'Female_speech_and_woman_speaking':4,
                           'Finger_snapping':5,
                           'Keys_jangling':6,
                           'Knock':7,
                           'Laughter':8,
                           'Male_speech_and_man_speaking':9,
                           'Printer':10,
                           'Scissors':11,
                           'Telephone':12,
                           'Writing':13}

def preprocessing_task1(args):
    '''
    predictors output: ambisonics mixture waveforms
                       Matrix shape: -x: data points
                                     -4 or 8: ambisonics channels
                                     -signal samples

    target output: monoaural clean speech waveforms
                   Matrix shape: -x: data points
                                 -1: it's monoaural
                                 -signal samples
    '''











def evaluate(model, device, criterion, dataloader):
    #compute loss without backprop
    model.eval()
    test_loss = 0.
    with tqdm(total=len(dataloader) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            target = target.to(device)
            x = x.to(device)
            outputs = model(x, device)
            loss = criterion(outputs, target)
            test_loss += (1. / float(example_num + 1)) * (loss - test_loss)
            pbar.set_description("Current val loss: {:.4f}".format(test_loss))
            pbar.update(1)
    return test_loss


def main(args):
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    if args.fixed_seed:
        seed = 1
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    #LOAD DATASET
    print ('\nLoading dataset')
    sr_task1 = 16000
    max_file_length_task1 = 12

    def pad(x, size):
        #pad all sounds to 4.792 seconds to meet the needs of Task1 baseline model MMUB
        length = x.shape[-1]
        if length > size:
            pad = x[:,:size]
        else:
            pad = np.zeros((x.shape[0], size))
            pad[:,:length] = x
        return pad

    def process_folder(folder, args):
        #process single dataset folder
        print ('Processing ' + folder + ' folder...')
        predictors = []
        target = []
        count = 0
        main_folder = os.path.join(args.input_path, folder)
        '''
        contents = os.listdir(main_folder)

        for sub in contents:
            sub_folder = os.path.join(main_folder, sub)
            contents_sub = os.listdir(sub_folder)
            for lower in contents_sub:
                lower_folder = os.path.join(sub_folder, lower)
                data_path = os.path.join(lower_folder, 'data')
                data = os.listdir(data_path)
                data = [i for i in data if i.split('.')[0].split('_')[-1]=='A']  #filter files with mic B
        '''
        data_path = os.path.join(main_folder, 'data')
        data = os.listdir(data_path)
        data = [i for i in data if i.split('.')[0].split('_')[-1]=='A']  #filter files with mic B
        for sound in data:
            sound_path = os.path.join(data_path, sound)
            sound_path = Path(sound_path).as_posix()
            target_path = '/'.join((sound_path.split('/')[:-2] + ['labels'] + [sound_path.split('/')[-1]]))  #change data with labels
            # print(target_path)
            target_path = target_path[:-6] + target_path[-4:]  #remove mic ID
            # print(data_path)
            # print(sound_path)
            #target_path = sound_path.replace('data', 'labels').replace('_A', '')  #old wrong line
            samples, sr = librosa.load(sound_path, sr_task1, mono=False)
            #samples = pad(samples)
            if args.num_mics == 2:  # if both ambisonics mics are wanted
                #stack the additional 4 channels to get a (8, samples) shap
                B_sound_path = sound_path[:-5] + 'B' +  sound_path[-4:]  #change A with B
                samples_B, sr = librosa.load(B_sound_path, sr_task1, mono=False)
                samples = np.concatenate((samples,samples_B), axis=-2)

            samples_target, sr = librosa.load(target_path, sr_task1, mono=False)
            samples_target = samples_target.reshape((1, samples_target.shape[0]))

            #append to final arrays
            if args.segmentation_len is not None:
                #segment longer file to shorter frames
                #not padding if segmenting to avoid silence frames
                segmentation_len_samps = int(sr_task1 * args.segmentation_len)
                predictors_cuts, target_cuts = uf.segment_waveforms(samples, samples_target, segmentation_len_samps)
                for i in range(len(predictors_cuts)):
                    predictors.append(predictors_cuts[i])
                    target.append(target_cuts[i])
            else:
                samples = pad(samples, size=int(sr_task1*args.pad_length))
                samples_target = pad(samples_target, size=int(sr_task1*args.pad_length))
                predictors.append(samples)
                target.append(samples_target)
            print ("here!!!! ", samples.shape)
            count += 1
            if args.num_data is not None and count >= args.num_data:
                break

        return predictors, target

    #process all required folders
    predictors_test, target_test = process_folder('L3DAS22_Task1_dev', args)

    if args.training_set == 'train100':
        predictors_train, target_train = process_folder('L3DAS22_Task1_train100', args)
    elif args.training_set == 'train360':
        predictors_train_1, target_train_1 = process_folder('L3DAS22_Task1_train360_1', args)
        predictors_train_2, target_train_2 = process_folder('L3DAS22_Task1_train360_2', args)
        predictors_train = predictors_train_1 + predictors_train_2
        target_train = target_train_1 + target_train_2
    elif args.training_set == 'both':
        predictors_train100, target_train100 = process_folder('L3DAS22_Task1_train100', args)
        predictors_train360_1, target_train360_1 = process_folder('L3DAS22_Task1_train360_1', args)
        predictors_train360_2, target_train360_2 = process_folder('L3DAS22_Task1_train360_2', args)
        predictors_train = predictors_train100 + predictors_train360_1 + predictors_train360_2
        target_train = target_train100 + target_train360_1 + target_train360_2

    #split train set into train and development
    split_point = int(len(predictors_train) * args.train_val_split)
    predictors_training = predictors_train[:split_point]    #attention: changed training names
    target_training = target_train[:split_point]
    predictors_validation = predictors_train[split_point:]
    target_validation = target_train[split_point:]


    # training_predictors = np.array(training_predictors)
    # training_target = np.array(training_target)
    # validation_predictors = np.array(validation_predictors)
    # validation_target = np.array(validation_target)
    # test_predictors = np.array(test_predictors)
    # test_target = np.array(test_target)

    # print ('\nShapes:')
    # print ('Training predictors: ', training_predictors.shape)
    # print ('Validation predictors: ', validation_predictors.shape)
    # print ('Test predictors: ', test_predictors.shape)

    #convert to tensor
    training_predictors = torch.tensor(predictors_training).float()
    validation_predictors = torch.tensor(predictors_validation).float()
    test_predictors = torch.tensor(predictors_test).float()
    training_target = torch.tensor(target_training).float()
    validation_target = torch.tensor(target_validation).float()
    test_target = torch.tensor(target_test).float()
    #build dataset from tensors
    tr_dataset = utils.TensorDataset(training_predictors, training_target)
    val_dataset = utils.TensorDataset(validation_predictors, validation_target)
    test_dataset = utils.TensorDataset(test_predictors, test_target)
    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)

    #LOAD MODEL
    if args.architecture == 'fasnet':
        model = FaSNet_origin(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                              hidden_dim=args.hidden_dim, layer=args.layer,
                              segment_size=args.segment_size, nspk=args.nspk,
                              win_len=args.win_len, context_len=args.context_len,
                              sr=args.sr)
    elif args.architecture == 'tac':
        model = FaSNet_TAC(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                              hidden_dim=args.hidden_dim, layer=args.layer,
                              segment_size=args.segment_size, nspk=args.nspk,
                              win_len=args.win_len, context_len=args.context_len,
                              sr=args.sr)
    elif args.architecture == 'MIMO_UNet_Beamforming':
        model = MIMO_UNet_Beamforming(fft_size=args.fft_size,
                                      hop_size=args.hop_size,
                                      input_channel=args.input_channel)
    elif args.architecture == 'MMULB':
        model = MMULB(fft_size=args.fft_size,
                      hop_size=args.hop_size,
                      input_channel=args.input_channel)

    if args.use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    #compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))

    #set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    #set up optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=0.00000001,
        verbose=True)


    #set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    #load model checkpoint if desired
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = load_model(model, optimizer, args.load_model, args.use_cuda)

    #TRAIN MODEL
    print('TRAINING START')
    train_loss_hist = []
    val_loss_hist = []
    epoch = 1
    while state["worse_epochs"] < args.patience:
        print("Training epoch " + str(epoch))
        avg_time = 0.
        model.train()
        train_loss = 0.
        with tqdm(total=len(tr_dataset) // args.batch_size) as pbar:
            for example_num, (x, target) in enumerate(tr_data):
                target = target.to(device)
                x = x.to(device)
                t = time.time()
                # Compute loss for each instrument/model
                optimizer.zero_grad()
                outputs = model(x, device)
                loss = criterion(outputs, target)
                loss.backward()

                train_loss += (1. / float(example_num + 1)) * (loss - train_loss)
                pbar.set_description("Current train loss: {:.4f}".format(train_loss))
                optimizer.step()
                state["step"] += 1
                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                pbar.update(1)

            #PASS VALIDATION DATA
            val_loss = evaluate(model, device, criterion, val_data)
            print("VALIDATION FINISHED: LOSS: " + str(val_loss))

            # EARLY STOPPING CHECK
            valid_loss = val_loss.cpu().detach().numpy()
            checkpoint_name = ('%03d' % epoch) + '_' + ('%.6f' % valid_loss) + '.pth'
            checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
            # checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint")

            if val_loss >= state["best_loss"]:
                state["worse_epochs"] += 1
                print(state["worse_epochs"])
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_loss"] = val_loss
                state["best_checkpoint"] = checkpoint_path

                # CHECKPOINT
                print("Saving model...")
                save_model(model, optimizer, state, checkpoint_path)

            state["epochs"] += 1
            #state["worse_epochs"] = 200
            train_loss_hist.append(train_loss.cpu().detach().numpy())
            val_loss_hist.append(val_loss.cpu().detach().numpy())
            epoch += 1
    #LOAD BEST MODEL AND COMPUTE LOSS FOR ALL SETS
    print("TESTING")
    # Load best model based on validation loss
    state = load_model(model, None, state["best_checkpoint"], args.use_cuda)
    #compute loss on all set_output_size
    train_loss = evaluate(model, device, criterion, tr_data)
    val_loss = evaluate(model, device, criterion, val_data)
    test_loss = evaluate(model, device, criterion, test_data)

    #PRINT AND SAVE RESULTS
    results = {'train_loss': train_loss.cpu().detach().numpy(),
               'val_loss': val_loss.cpu().detach().numpy(),
               'test_loss': test_loss.cpu().detach().numpy(),
               'train_loss_hist': train_loss_hist,
               'val_loss_hist': val_loss_hist}

    print ('RESULTS')
    for i in results:
        if 'hist' not in i:
            print (i, results[i])
    out_path = os.path.join(args.results_path, 'results_dict.json')
    np.save(out_path, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        # i/o
        parser.add_argument('--task', type=int, default=1,
                            help='task to be pre-processed')
        parser.add_argument('--input_path', type=str, default='H:/dataset/2022L3DSE',
                            help='directory where the dataset has been downloaded')
        parser.add_argument('--output_path', type=str, default='DATASETS/processed',
                            help='where to save the numpy matrices')
        # processing type
        parser.add_argument('--train_val_split', type=float, default=0.7,
                            help='perc split between train and validation sets')
        parser.add_argument('--num_mics', type=int, default=2,
                            help='how many ambisonics mics (1 or 2)')
        parser.add_argument('--num_data', type=int, default=None,
                            help='how many datapoints per set. 0 means all available data')
        # task1 only parameters
        # the following parameters produce 2-seconds waveform frames without overlap,
        # use only the train100 training set.
        parser.add_argument('--training_set', type=str, default='train100',
                            help='which training set: train100, train360 or both')
        parser.add_argument('--segmentation_len', type=float, default=1.912,
                            help='length of segmented frames in seconds')
        # task2 only parameters
        # the following stft parameters produce 8 stft fframes per each label frame
        # if label frames are 100msecs, stft frames are 12.5 msecs
        # data-points are segmented into 15-seconde windows (150 target frames, 150*8 stft frames)
        parser.add_argument('--frame_len', type=int, default=100,
                            help='frame length for SELD evaluation (in msecs)')
        parser.add_argument('--stft_nperseg', type=int, default=512,
                            help='num of stft frames')
        parser.add_argument('--stft_noverlap', type=int, default=112,
                            help='num of overlapping samples for stft')
        parser.add_argument('--stft_window', type=str, default='hamming',
                            help='stft window_type')
        parser.add_argument('--output_phase', type=str, default='False',
                            help='concatenate phase channels to stft matrix')
        parser.add_argument('--predictors_len_segment', type=int, default=None,
                            help='number of segmented frames for stft data')
        parser.add_argument('--target_len_segment', type=int, default=None,
                            help='number of segmented frames for stft data')
        parser.add_argument('--segment_overlap', type=float, default=None,
                            help='overlap factor for segmentation')
        parser.add_argument('--pad_length', type=float, default=4.792,
                            help='length of signal padding in seconds')
        parser.add_argument('--ov_subsets', type=str, default='["ov1", "ov2", "ov3"]',
                            help='should be a list of strings. Can contain ov1, ov2 and/or ov3')
        parser.add_argument('--no_overlaps', type=str, default='False',
                            help='should be a list of strings. Can contain ov1, ov2 and/or ov3')

    #saving parameters
    parser.add_argument('--results_path', type=str, default='RESULTS/2MMUBTask1',
                        help='Folder to write results dicts into')
    parser.add_argument('--checkpoint_dir', type=str, default='RESULTS/2MMUBTask1',
                        help='Folder to write checkpoints into')
    #dataset parameters
    # parser.add_argument('--training_predictors_path', type=str, default='/kaggle/input/processed100pad-dev/processed100pad_dev/task1_predictors_train.pkl')
    # parser.add_argument('--training_target_path', type=str, default='/kaggle/input/processed100pad-dev/processed100pad_dev/task1_target_train.pkl')
    # parser.add_argument('--validation_predictors_path', type=str, default='/kaggle/input/processed100pad-dev/processed100pad_dev/task1_predictors_validation.pkl')
    # parser.add_argument('--validation_target_path', type=str, default='/kaggle/input/processed100pad-dev/processed100pad_dev/task1_target_validation.pkl')
    # parser.add_argument('--test_predictors_path', type=str, default='/kaggle/input/processed100pad-dev/processed100pad_dev/task1_predictors_test.pkl')
    # parser.add_argument('--test_target_path', type=str, default='/kaggle/input/processed100pad-dev/processed100pad_dev/task1_target_test.pkl')
    #training parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--early_stopping', type=str, default='True')
    parser.add_argument('--fixed_seed', type=str, default='False')
    parser.add_argument('--load_model', type=str, default='RESULTS/2MMUBTask1/049_0.023416.pth',
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=3,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument('--patience', type=int, default=5,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    #model parameters
    parser.add_argument('--architecture', type=str, default='MMULB',
                        help="model name")
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

    #eval string bools
    args.use_cuda = eval(args.use_cuda)
    args.early_stopping = eval(args.early_stopping)
    args.fixed_seed = eval(args.fixed_seed)

    main(args)
