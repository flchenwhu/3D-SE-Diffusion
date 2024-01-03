import matplotlib.pyplot as plt
import wave
import librosa.display
import numpy as np


# path = "H:/dataset/2022L3DSE/L3DAS22_Task1_test/labels/1089-134686-0000.wav"
# path ="RESULTS/MMUBTask1/dev/sounds/800_cut.wav"
# path="video/sound/dev/MSWB1mic.wav"
path="F:/DMSE/resultsclean/sgmselast/enhance/dev/noisy/8.wav"
# path = "RESULTS/MSWBTask1/test_evaluate/sounds/0_cut.wav"
def plotspectorgeam(path):
    f = wave.open(path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # print("声道数---", nchannels)
    # print("量化位数---", sampwidth)
    # print("采样频率---", framerate)
    # print("采样点数---", nframes)
    data = f.readframes(nframes)

    # 归一化
    data = np.fromstring(data, dtype=np.int16)
    data = data * 1.0 / (max(abs(data)))
    data = np.reshape(data, [nframes, nchannels]).T
    # data = data * 1.0 / max(data)
    f.close()
    # 32ms
    framelength = 0.032
    # NFFT点数=0.032*fs
    framesize = int(framelength * framerate)
    # print("NFFT:", framesize)

    # 画语谱图
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.specgram(data[0], NFFT=framesize, Fs=framerate, window=np.hanning(M=framesize))
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    # plt.title('Spectrogram')
    plt.show()

def plotMel(path):
    f = wave.open(path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print("声道数---", nchannels)
    print("量化位数---", sampwidth)
    print("采样频率---", framerate)
    print("采样点数---", nframes)
    data = f.readframes(nframes)

    # 归一化
    data = np.fromstring(data, dtype=np.int16)
    data = data * 1.0 / (max(abs(data)))
    data = np.reshape(data, [nframes, nchannels]).T
    # data = data * 1.0 / max(data)
    f.close()

    # 0.032s
    framelength = 0.032
    # NFFT点数=0.032*fs
    framesize = int(framelength * framerate)
    print("NFFT:", framesize)

    # 提取mel特征
    mel_spect = librosa.feature.melspectrogram(data[0], sr=framerate, n_fft=framesize)
    # 转化为log形式
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    # 画mel谱图
    # librosa.display.specshow(mel_spect, sr=framerate)
    librosa.display.specshow(mel_spect, sr=framerate, x_axis='time', y_axis='mel')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')
    plt.show()

# plotspectorgeam(path)
plotMel(path)



