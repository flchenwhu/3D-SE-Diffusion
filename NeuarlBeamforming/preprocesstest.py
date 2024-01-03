import argparse
import os
import pickle
import random
import sys

import librosa
import numpy as np

import utility_functions as uf
from pathlib import Path

sr_task1 = 16000
max_file_length_task1 = 12

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

    # predictors_test, target_test = process_folder('L3DAS22_Task1_dev', args)
    # if args.training_set == 'train100':
    #     predictors_train, target_train = process_folder('L3DAS22_Task1_train100', args)
    # elif args.training_set == 'train360':
    #     predictors_train_1, target_train_1 = process_folder('L3DAS22_Task1_train360_1', args)
    #     predictors_train_2, target_train_2 = process_folder('L3DAS22_Task1_train360_2', args)
    #     predictors_train = predictors_train_1 + predictors_train_2
    #     target_train = target_train_1 + target_train_2
    # elif args.training_set == 'both':
    #     predictors_train100, target_train100 = process_folder('L3DAS22_Task1_train100', args)
    #     predictors_train360_1, target_train360_1 = process_folder('L3DAS22_Task1_train360_1', args)
    #     predictors_train360_2, target_train360_2 = process_folder('L3DAS22_Task1_train360_2', args)
    #     predictors_train = predictors_train100 + predictors_train360_1 + predictors_train360_2
    #     target_train = target_train100 + target_train360_1 + target_train360_2
    #
    #     # split train set into train and development
    # split_point = int(len(predictors_train) * args.train_val_split)
    # predictors_training = predictors_train[:split_point]  # attention: changed training names
    # target_training = target_train[:split_point]
    # predictors_validation = predictors_train[split_point:]
    # target_validation = target_train[split_point:]
    #
    # # save numpy matrices in pickle files
    # print('Saving files')
    # if not os.path.isdir(args.output_path):
    #     os.makedirs(args.output_path)
    #
    # with open(os.path.join(args.output_path, 'task1_predictors_train.pkl'), 'wb') as f:
    #     pickle.dump(predictors_training, f, protocol=4)
    # with open(os.path.join(args.output_path, 'task1_predictors_validation.pkl'), 'wb') as f:
    #     pickle.dump(predictors_validation, f, protocol=4)
    # with open(os.path.join(args.output_path, 'task1_target_train.pkl'), 'wb') as f:
    #     pickle.dump(target_training, f, protocol=4)
    # with open(os.path.join(args.output_path, 'task1_target_validation.pkl'), 'wb') as f:
    #     pickle.dump(target_validation, f, protocol=4)
    # with open(os.path.join(args.output_path,'task1_predictors_test.pkl'), 'wb') as f:
    #     pickle.dump(predictors_test, f, protocol=4)
    # with open(os.path.join(args.output_path,'task1_target_test.pkl'), 'wb') as f:
    #     pickle.dump(target_test, f, protocol=4)
    #
    # print ('Matrices successfully saved')
    # print ('Training set shape: ', np.array(predictors_training).shape, np.array(target_training).shape)
    # print ('Validation set shape: ', np.array(predictors_validation).shape, np.array(target_validation).shape)
    # print ('lossTest set shape: ', np.array(predictors_test).shape, np.array(target_test).shape)


    #generate also a test set matrix with full-length samples, just for the evaluation
    print ('processing uncut test set')
    args.pad_length = max_file_length_task1  #only here use max_file_length
    predictors_test_uncut, target_test_uncut = process_folder('L3DAS22_Task1_dev', args)
    print ('Saving files')
    with open(os.path.join(args.output_path,'task1_predictors_test_uncut.pkl'), 'wb') as f:
        pickle.dump(predictors_test_uncut, f)
    with open(os.path.join(args.output_path,'task1_target_test_uncut.pkl'), 'wb') as f:
        pickle.dump(target_test_uncut, f)


    # print ('Test set shape: ', np.array(predictors_test_uncut).shape, np.array(target_test_uncut).shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #i/o
    parser.add_argument('--task', type=int, default=1,
                        help='task to be pre-processed')
    parser.add_argument('--input_path', type=str, default='H:/dataset/2022L3DSE',
                        help='directory where the dataset has been downloaded')
    parser.add_argument('--output_path', type=str, default='DATASETS/processedcut1.912A',
                        help='where to save the numpy matrices')
    #processing type
    parser.add_argument('--train_val_split', type=float, default=0.7,
                        help='perc split between train and validation sets')
    parser.add_argument('--num_mics', type=int, default=1,
                        help='how many ambisonics mics (1 or 2)')
    parser.add_argument('--num_data', type=int, default=None,
                        help='how many datapoints per set. 0 means all available data')
    #task1 only parameters
    #the following parameters produce 2-seconds waveform frames without overlap,
    #use only the train100 training set.
    parser.add_argument('--training_set', type=str, default='train100',
                        help='which training set: train100, train360 or both')
    parser.add_argument('--segmentation_len', type=float, default=None,
                        help='length of segmented frames in seconds')
    #task2 only parameters
    #the following stft parameters produce 8 stft fframes per each label frame
    #if label frames are 100msecs, stft frames are 12.5 msecs
    #data-points are segmented into 15-seconde windows (150 target frames, 150*8 stft frames)
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
    parser.add_argument('--pad_length', type=float, default=1.912,
                        help='length of signal padding in seconds')
    parser.add_argument('--ov_subsets', type=str, default='["ov1", "ov2", "ov3"]',
                        help='should be a list of strings. Can contain ov1, ov2 and/or ov3')
    parser.add_argument('--no_overlaps', type=str, default='False',
                        help='should be a list of strings. Can contain ov1, ov2 and/or ov3')


    args = parser.parse_args()

    args.output_phase = eval(args.output_phase)
    args.ov_subsets = eval(args.ov_subsets)
    args.no_overlaps = eval(args.no_overlaps)

    if args.task == 1:
        preprocessing_task1(args)
