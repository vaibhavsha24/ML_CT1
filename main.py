import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from glob import glob
from librosa import feature
import numpy as np
#directories of normal audios
# Press the green button in the gutter to run the script.



fn_list_i = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff
]

fn_list_ii = [
    feature.rms,
    feature.zero_crossing_rate
]

def get_feature_vector(y, sr):
    feat_vect_i = [np.mean(funct(y, sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y)) for funct in fn_list_ii]
    feature_vector = feat_vect_i + feat_vect_ii
    return feature_vector


if __name__ == '__main__':
    # audio_file_path="D:\\1st sem\\ML\\CT\\Actor_01\\03-01-01-01-02-02-01.wav"
    # librosa_audio_data,librosa_sample_rate=librosa.load(audio_file_path)
    # print(librosa_audio_data)
    # plt.figure(figsize=(12,4))
    # plt.plot(librosa_audio_data)
    # plt.show()
    # wave_sample_rate, wave_audio = wav.read(audio_file_path)
    # wave_audio
    norm_data_dir = "D:\\1st sem\\ML\\CT\\Actor_01\\"
    norm_audio_files = glob(norm_data_dir + '*.wav')
    norm_audios_feat = []
    for file in norm_audio_files:
        y, sr = librosa.load(file, sr=None)
        feature_vector = get_feature_vector(y, sr)
        norm_audios_feat.append(feature_vector)
    print(norm_audios_feat)
