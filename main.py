import librosa
import matplotlib.pyplot as plt
from glob import glob
from librosa import feature
import numpy as np
import csv

#directories of normal audios
# Press the green button in the gutter to run the script.



fn_list_i = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
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
    norm_audio_list=[]
    for i in range(1,24):
        actor_dir="Actor_0"+str(i)
        if(i>=10):
            actor_dir="Actor_"+str(i)
        norm_data_dir = "C:\\Users\\Admin\\PycharmProjects\\ML_CT1\\venv\\files\\{a}\\".format(a=actor_dir)
        norm_audio_files = glob(norm_data_dir + '*.wav')
        norm_audios_feat = []
        for file in norm_audio_files:
            y, sr = librosa.load(file, sr=None)
            feature_vector = get_feature_vector(y, sr)
            norm_audios_feat.append(feature_vector)
        norm_audio_list.append(norm_audios_feat)
    print(norm_audio_list)

    norm_output = 'normals_00.csv'
    header = [
    'chroma_stft',
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_rolloff',
    'rmse',
    'zero_crossing_rate'
    ]
    with open(norm_output,'+w') as f:
        csv_writer = csv.writer(f, delimiter= ',' )
        csv_writer.writerow(header)
        csv_writer.writerows(norm_audio_list)


