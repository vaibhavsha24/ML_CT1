import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    audio_file_path="D:\\1st sem\\ML\\CT\\Actor_01\\03-01-01-01-02-02-01.wav"
    librosa_audio_data,librosa_sample_rate=librosa.load(audio_file_path)
    print(librosa_audio_data)
    plt.figure(figsize=(12,4))
    plt.plot(librosa_audio_data)
    plt.show()
    wave_sample_rate, wave_audio = wav.read(audio_file_path)
    wave_audio
